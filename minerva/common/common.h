#pragma once
#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <cstdio>
#include <unordered_set>
#include <algorithm>
#include "./make_unique.h"

#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete; \
  TypeName& operator=(const TypeName&) = delete

#define DISALLOW_MOVE_AND_ASSIGN(TypeName) \
  TypeName(TypeName&&) = delete; \
  TypeName& operator=(TypeName&&) = delete

#define DISALLOW_COPY_AND_MOVE(TypeName) \
  DISALLOW_COPY_AND_ASSIGN(TypeName); \
  DISALLOW_MOVE_AND_ASSIGN(TypeName)


/*
 *  Serialization
 */
#define SERIALIZE(Buff, Off, Item, Type) \
	*(reinterpret_cast<Type*>(Buff+Off)) = Item; \
	Off += sizeof(Type);

#define DESERIALIZE(Buff, Off, Item, Type) \
	Item = *(reinterpret_cast<Type *>(Buff+Off)); \
	Off += sizeof(Type);

#define DESERIALIZE_ENUM(Buff, Off, Item, Type) \
	Item = static_cast<Type>(*(Buff+Off)); \
	Off += sizeof(int);


# if defined(_MSC_VER)
# define __attribute__(A) /* do nothing */
# endif


namespace minerva {

class Serializable{
public:
	virtual ~Serializable() {};
	virtual int GetSerializedSize() const =0;
	virtual int Serialize(char*) const =0;
};


template<typename T>
std::ostream& operator<<(std::ostream& os, const std::set<T>& s) {
  os << "{ ";
  for (const T& t: s) {
    os << t << " ";
  }
  return os << "}";
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::unordered_set<T>& s) {
  os << "{ ";
  for (const T& t: s) {
    os << t << " ";
  }
  return os << "}";
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
  os << "[ ";
  for (const T& t: v) {
    os << t << " ";
  }
  return os << "]";
}

template<typename U, typename T, typename Fn>
std::vector<U> Map(const std::vector<T>& original, Fn fn) {
  std::vector<U> res;
  res.resize(original.size());
  std::transform(original.begin(), original.end(), res.begin(), fn);
  return res;
}

template<typename Iterable, typename Fn>
void Iter(const Iterable& original, Fn fn) {
  std::for_each(original.begin(), original.end(), fn);
}

namespace common {

template<typename... Args>
std::string FString(char const* format, Args&&... args) {
# if defined(_MSC_VER)
    const size_t buffer_size = 1024;
# else
    size_t constexpr buffer_size = 1024;
# endif
  char buffer[buffer_size];
  snprintf(buffer, buffer_size, format, std::forward<Args>(args)...);
  return std::string(buffer);
}

void FatalError(char const* format, ...)
    __attribute__((format(printf, 1, 2), noreturn));

}  // namespace common
}  // namespace minerva

