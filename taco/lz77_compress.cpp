#include "png_reader.h"

// 0 XXXXXXX XXXXXXXX     -> read X number of bytes
// 1 XXXXXXX XXXXXXXX Y Y -> X is the run length, Y is the distance

#define MAX_LIT  32767
#define MAX_RUN  32767
#define MAX_DIST 65535

int computeHash(const std::vector<uint8_t>& in, int in_idx){
  return ( ( in[ in_idx ] & 0xFF ) << 8 ) |
         ( ( in[ in_idx + 1 ] & 0xFF ) ^ ( in[ in_idx + 2 ] & 0xFF ) );
}

std::pair<int,int> find_match(const int in_idx, const int out_idx, const int start_idx,
                              const std::vector<uint8_t>& in, const std::vector<uint8_t>& out) {
  int len = 1;
  int off = out_idx - start_idx;
  if( off > 0 && off < 65536 && start_idx >= 0 && start_idx < out.size() &&
      out[ start_idx ] == in[ in_idx ] ) {
    while( in_idx + len < in.size() &&
           out[ start_idx + (len % off) ] == in[ in_idx + len ] ) {
      len++;
    }
  }
  return {len, off};
}

union short_byte {
    uint16_t val;
    uint8_t bytes[2];
};

void push_uint16(std::vector<uint8_t>& out, uint16_t value){
  short_byte store{value};
  out.push_back(store.bytes[0]);
  out.push_back(store.bytes[1]);
}

void set_uint16(std::vector<uint8_t>& out, int index, uint16_t value){
  short_byte store{value};
  out[index + 0] = store.bytes[0];
  out[index + 1] = store.bytes[1];
}

uint16_t load_uint16(std::vector<uint8_t>& out, int index){
  short_byte store;
  store.bytes[0] = out[index];
  store.bytes[1] = out[index+1];
  return store.val;
}

void push_one(std::vector<uint8_t>& out, const std::vector<uint8_t>& in, std::vector<int>& hash, int& count_idx, int& in_idx){
  int len = 1;
  if (count_idx != -1 && load_uint16(out, count_idx)==MAX_LIT){
    count_idx = -1;
  }
  if (count_idx == -1) {
    count_idx = out.size();
    push_uint16(out, 0);
  }

  int old_cout = load_uint16(out, count_idx);

  int new_count = len+old_cout > MAX_LIT ? MAX_LIT : len+old_cout;
  int max = new_count - old_cout;
  len -= max;
  set_uint16(out, count_idx, new_count);
  if (count_idx+2<out.size()) {
    hash[computeHash(out, count_idx)] = count_idx;
  }
  while( max ) {
    out.push_back(in[ in_idx++ ]);
    if(out.size() > 2) {
      hash[computeHash(out, out.size()-3)] = out.size()-3;
    }
    max--;
  }
}

std::pair<std::vector<uint8_t>, int> encode_lz77(const std::vector<uint8_t> in) {
  int in_idx = 0;
  int count_idx = -1;
  std::vector<int> hash(65536);
  int numRaw = 0;

  std::vector<uint8_t> out;

  while( in_idx < in.size() ) {
    int len = 1;
    int off = 0;
    if( in_idx + 2 < in.size() ) {
      auto rle = find_match(in_idx, out.size(), out.size() - 1, in, out);
      auto pixelRle = find_match(in_idx, out.size(), out.size() - 3, in, out);
      auto hashCheck = find_match(in_idx, out.size(), hash[ computeHash(in, in_idx) ], in, out);
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
      push_uint16(out, 32768 | len);
      push_uint16(out, off);
      count_idx = -1;
      hash[computeHash(out, out.size()-4)] = out.size()-4;
      hash[computeHash(out, out.size()-3)] = out.size()-3;
      in_idx+=len;
      if (in_idx < in.size()){
        push_one(out,in,hash,count_idx,in_idx);
        numRaw++;
      }
    } else 
    {
      while( len ) {
        if (count_idx != -1 && load_uint16(out, count_idx)==MAX_LIT){
          count_idx = -1;
        }
        if (count_idx == -1) {
          count_idx = out.size();
          push_uint16(out, 0);
        }
        
        int old_cout = load_uint16(out, count_idx);

        int new_count = len+old_cout > MAX_LIT ? MAX_LIT : len+old_cout;
        int max = new_count - old_cout;
        len -= max;
        set_uint16(out, count_idx, new_count);
        if (count_idx+2<out.size()) {
          hash[computeHash(out, count_idx)] = count_idx;
        }
        while( max ) {
          out.push_back(in[ in_idx++ ]);
          numRaw++;
          if(out.size() > 2) {
            hash[computeHash(out, out.size()-3)] = out.size()-3;
          }
          max--;
        }
      }
    }
  }
  return {out, numRaw};
}