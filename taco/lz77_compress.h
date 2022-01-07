#include "png_reader.h"

// 0 XXXXXXX XXXXXXXX     -> read X number of bytes
// 1 XXXXXXX XXXXXXXX Y Y -> X is the run length, Y is the distance

#define MAX_LIT  32767
#define MAX_RUN  32767
#define MAX_DIST 65535

template <class T>
union value_bytes {
    T val;
    uint8_t bytes[sizeof(T)];
};

template <class T>
T load_type(const uint8_t* out, int index){
  value_bytes<T> v;
  for (int i = 0; i < sizeof(T); i++){
    v.bytes[i] = out[index + i];
  }
  return v.val;
}

template <class T>
void store_type(uint8_t* out, int index, T value){
  value_bytes<T> v{value};
  for (int i = 0; i < sizeof(T); i++){
    out[index + i] = v.bytes[i];
  }
}

template <class T>
void push_type(std::vector<uint8_t>& out, T value){
  value_bytes<T> v{value};
  for (int i = 0; i < sizeof(T); i++){
    out.push_back(v.bytes[i]);
  }
}

inline int computeHash(const uint8_t* in, int in_idx){
  return ( ( in[ in_idx ] & 0xFF ) << 8 ) |
         ( ( in[ in_idx + 1 ] & 0xFF ) ^ ( in[ in_idx + 2 ] & 0xFF ) );
}

template <class T>
std::pair<int,int> find_match(const int in_idx_, const int out_idx, const int start_idx,
                              const std::vector<T>& in, const std::vector<uint8_t>& out) {
  int len = sizeof(T);
  int off = out_idx - start_idx;

  uint8_t* in_data = (uint8_t*) &in[0];
  int in_idx = in_idx_ * sizeof(T);
  size_t in_size = in.size() * sizeof(T);

  if( off > 0 && off < 65536 && start_idx >= 0 && start_idx < out.size() && start_idx + sizeof(T) <= out.size() &&
      load_type<T>(&out[0], start_idx) == load_type<T>(in_data, in_idx) ) {
        if (off % sizeof(T) == 0 ){
          while( in_idx + len/sizeof(T) < in_size &&
                load_type<T>(&out[0], start_idx + (len % off)) == load_type<T>(in_data, in_idx + len) ) {
            len+=sizeof(T);
          }
        } else {
          while( in_idx + len/sizeof(T) < in_size && start_idx + (len % off) + sizeof(T) < out.size() &&
                load_type<T>(&out[0], start_idx + (len % off)) == load_type<T>(in_data, in_idx + len) ) {

            len+=sizeof(T);
          }
        }
  }
  return {len/sizeof(T), off};
}

template <class T>
void push_one(std::vector<uint8_t>& out, const std::vector<T>& in, std::vector<int>& hash, int& count_idx, int& in_idx){
  int len = 1;
  if (count_idx != -1 && load_type<uint16_t>(&out[0], count_idx)==MAX_LIT){
    count_idx = -1;
  }
  if (count_idx == -1) {
    count_idx = out.size();
    push_type<uint16_t>(out, 0);
  }

  int old_count = load_type<uint16_t>(&out[0], count_idx);

  int new_count = len+old_count > MAX_LIT ? MAX_LIT : len+old_count;
  int max = new_count - old_count;
  len -= max;
  store_type<uint16_t>(&out[0], count_idx, new_count);
  if (count_idx+2<out.size()) {
    hash[computeHash(&out[0], count_idx)] = count_idx;
  }
  while( max ) {
    push_type<T>(out, in[in_idx++]);
    for (int i = out.size() - 3; i > std::max((size_t) 0,out.size() - 3 - sizeof(T)); i-- ){
      hash[computeHash(&out[0], i)] = i;
    }
    max--;
  }
}

template <class T>
std::pair<std::vector<uint8_t>, int> encode_lz77(const std::vector<T> in) {
  int in_idx = 0;
  int count_idx = -1;
  std::vector<int> hash(65536);
  int numRaw = 0;

  std::vector<uint8_t> out;

  while( in_idx < in.size() ) {
    // std::cout << "in_idx " << in_idx << std::endl;
    int len = 1;
    int off = 0;
    if( in_idx + 2 < in.size() ) {
      auto rle = find_match<T>(in_idx, out.size(), out.size() - sizeof(T), in, out);
      auto pixelRle = find_match<T>(in_idx, out.size(), out.size() - 3*sizeof(T), in, out);
      auto hashCheck = find_match<T>(in_idx, out.size(), hash[ computeHash((uint8_t*)&in[0], in_idx*sizeof(T)) ], in, out);
      len = std::max({rle.first, pixelRle.first, hashCheck.first});
      if (rle.first == len){
        off = rle.second;
      } else if (pixelRle.first == len) {
        off = pixelRle.second;
      } else {
        off = hashCheck.second;
      }
    }
    if( len >= 4 ) {
      if( len > MAX_LIT ) {
        len = MAX_LIT;
      }
      push_type<uint16_t>(out, 32768 | len);
      push_type<uint16_t>(out, off);
      count_idx = -1;
      hash[computeHash(&out[0], out.size()-4)] = out.size()-4;
      hash[computeHash(&out[0], out.size()-3)] = out.size()-3;
      in_idx+=len;
      if (in_idx < in.size()){
        push_one(out,in,hash,count_idx,in_idx);
        numRaw++;
      }
    } else 
    {
      while( len ) {
        if (count_idx != -1 && load_type<uint16_t>(&out[0], count_idx)==MAX_LIT){
          count_idx = -1;
        }
        if (count_idx == -1) {
          count_idx = out.size();
          push_type<uint16_t>(out, 0);
        }
        
        int old_count = load_type<uint16_t>(&out[0], count_idx);

        int new_count = len+old_count > MAX_LIT ? MAX_LIT : len+old_count;
        int max = new_count - old_count;
        len -= max;
        store_type<uint16_t>(&out[0], count_idx, new_count);
        if (count_idx+2<out.size()) {
          hash[computeHash(&out[0], count_idx)] = count_idx;
        }
        while( max ) {
          push_type<T>(out, in[in_idx++]);
          numRaw++;
          for (int i = out.size() - 3; i > std::max((size_t) 0,out.size() - 3 - sizeof(T)); i-- ){
            hash[computeHash(&out[0], i)] = i;
          }
          max--;
        }
      }
    }
  }
  return {out, numRaw};
}