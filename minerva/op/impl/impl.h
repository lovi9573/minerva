#pragma once

namespace minerva {

template<typename C>
class FnBundle {
};

#ifdef HAS_CUDA
#ifdef HAS_MPI


#define INSTALL_COMPUTE_FN(closure_name, basic_fn, mkl_fn, cuda_fn, mpi_fn) \
  template<> class FnBundle<closure_name> {\
   public:\
    static void Call(const DataList& i, const DataList& o, closure_name& c, const Context& context) {\
      switch (context.impl_type) {\
        case ImplType::kBasic: basic_fn(i, o, c); break;\
        case ImplType::kMkl: mkl_fn(i, o, c); break;\
        case ImplType::kCuda: cuda_fn(i, o, c, context); break; \
        default: NO_IMPL(i, o, c, context); break;\
      }\
    }\
	\
	static void Call(const Task& task, closure_name& c, const Context& context) {\
	  switch (context.impl_type) {\
		case ImplType::kMpi: mpi_fn(task,c,context); break; \
		default: NO_IMPL(task, c, context); break;\
	  }\
	}\
	static int GetSerializedSize(){ \
		return closure_name.GetSerializedSize(); \
	} \
  };

#define INSTALL_DATAGEN_FN(closure_name, basic_fn, mkl_fn, cuda_fn, mpi_fn) \
  template<> class FnBundle<closure_name> {\
   public:\
    static void Call(const DataList& d, closure_name& c, const Context& context) {\
      switch (context.impl_type) {\
        case ImplType::kBasic: basic_fn(d, c); break;\
        case ImplType::kMkl: mkl_fn(d, c); break;\
        case ImplType::kCuda: cuda_fn(d, c, context); break;\
        default: NO_IMPL(d, c, context); break;\
      }\
    }\
	\
	static void Call(const Task& task, closure_name& c, const Context& context) {\
	  switch (context.impl_type) {\
		case ImplType::kMpi: mpi_fn(task,c,context); break; \
		default: NO_IMPL(task, c, context); break;\
	  }\
	}\
  };


#else //CUDA w/out MPI

#define INSTALL_COMPUTE_FN(closure_name, basic_fn, mkl_fn, cuda_fn, mpi_fn) \
  template<> class FnBundle<closure_name> {\
   public:\
    static void Call(const DataList& i, const DataList& o, closure_name& c, const Context& context) {\
      switch (context.impl_type) {\
        case ImplType::kBasic: basic_fn(i, o, c); break;\
        case ImplType::kMkl: mkl_fn(i, o, c); break;\
        case ImplType::kCuda: cuda_fn(i, o, c, context); break; \
        default: NO_IMPL(i, o, c, context); break;\
      }\
    }\
  };

#define INSTALL_DATAGEN_FN(closure_name, basic_fn, mkl_fn, cuda_fn, mpi_fn) \
  template<> class FnBundle<closure_name> {\
   public:\
    static void Call(const DataList& d, closure_name& c, const Context& context) {\
      switch (context.impl_type) {\
        case ImplType::kBasic: basic_fn(d, c); break;\
        case ImplType::kMkl: mkl_fn(d, c); break;\
        case ImplType::kCuda: cuda_fn(d, c, context); break;\
        default: NO_IMPL(d, c, context); break;\
      }\
    }\
  };
#endif//end CUDA -> ?MPI?

#else //no CUDA

#ifdef HAS_MPI

#define INSTALL_COMPUTE_FN(closure_name, basic_fn, mkl_fn, cuda_fn, mpi_fn) \
  template<> class FnBundle<closure_name> {\
   public:\
    static void Call(const DataList& i, const DataList& o, closure_name& c, const Context& context) {\
      switch (context.impl_type) {\
        case ImplType::kBasic: basic_fn(i, o, c); break;\
        case ImplType::kMkl: mkl_fn(i, o, c); break;\
        default: NO_IMPL(i, o, c, context); break;\
      }\
    }\
	\
	static void Call(const Task& task, closure_name& c, const Context& context) {\
	  switch (context.impl_type) {\
		case ImplType::kMpi: mpi_fn(task,c,context); break; \
		default: NO_IMPL(task, c, context); break;\
	  }\
	}\
  };

#define INSTALL_DATAGEN_FN(closure_name, basic_fn, mkl_fn, cuda_fn, mpi_fn) \
  template<> class FnBundle<closure_name> {\
   public:\
    static void Call(const DataList& d, closure_name& c, const Context& context) {\
      switch (context.impl_type) {\
        case ImplType::kBasic: basic_fn(d, c); break;\
        case ImplType::kMkl: mkl_fn(d, c); break;\
        default: NO_IMPL(d, c, context); break;\
      }\
    }\
	\
	static void Call(const Task& task, closure_name& c, const Context& context) {\
	  switch (context.impl_type) {\
		case ImplType::kMpi: mpi_fn(task,c,context); break; \
		default: NO_IMPL(task, c, context); break;\
	  }\
	}\
  };

#else //no MPI nor CUDA



#define INSTALL_COMPUTE_FN(closure_name, basic_fn, mkl_fn, cuda_fn, mpi_fn) \
  template<> class FnBundle<closure_name> {\
   public:\
    static void Call(const DataList& i, const DataList& o, closure_name& c, const Context& context) {\
      switch (context.impl_type) {\
        case ImplType::kBasic: basic_fn(i, o, c); break;\
        case ImplType::kMkl: mkl_fn(i, o, c); break;\
        default: NO_IMPL(i, o, c, context); break;\
      }\
    }\
  };

#define INSTALL_DATAGEN_FN(closure_name, basic_fn, mkl_fn, cuda_fn, mpi_fn) \
  template<> class FnBundle<closure_name> {\
   public:\
    static void Call(const DataList& d, closure_name& c, const Context& context) {\
      switch (context.impl_type) {\
        case ImplType::kBasic: basic_fn(d, c); break;\
        case ImplType::kMkl: mkl_fn(d, c); break;\
        default: NO_IMPL(d, c, context); break;\
      }\
    }\
  };

#endif //end ?MPI?
#endif //end ?CUDA?

}
