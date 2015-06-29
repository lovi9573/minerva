#pragma once

namespace minerva {

template<typename C>
class FnBundle {
};


#ifdef HAS_MPI

	#if defined HAS_CUDA && defined HAS_FPGA

		#define INSTALL_COMPUTE_FN(closure_name, basic_fn, mkl_fn, cuda_fn, fpga_fn, mpi_fn) \
		  template<> class FnBundle<closure_name> {\
		   public:\
			static void Call(const DataList& i, const DataList& o, closure_name& c, const Context& context) {\
			  switch (context.impl_type) {\
				case ImplType::kBasic: basic_fn(i, o, c); break;\
				case ImplType::kMkl: mkl_fn(i, o, c); break;\
				case ImplType::kCuda: cuda_fn(i, o, c, context); break; \
				case ImplType::kFpga: fpga_fn(i, o, c, context); break; \
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

		#define INSTALL_DATAGEN_FN(closure_name, basic_fn, mkl_fn, cuda_fn, fpga_fn, mpi_fn) \
		  template<> class FnBundle<closure_name> {\
		   public:\
			static void Call(const DataList& d, closure_name& c, const Context& context) {\
			  switch (context.impl_type) {\
				case ImplType::kBasic: basic_fn(d, c); break;\
				case ImplType::kMkl: mkl_fn(d, c); break;\
				case ImplType::kCuda: cuda_fn(d, c, context); break;\
				case ImplType::kFpga: fpga_fn(d, c, context); break; \
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


	#elif defined HAS_CUDA

		#define INSTALL_COMPUTE_FN(closure_name, basic_fn, mkl_fn, cuda_fn, fpga_fn, mpi_fn) \
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
		  };

		#define INSTALL_DATAGEN_FN(closure_name, basic_fn, mkl_fn, cuda_fn, fpga_fn, mpi_fn) \
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

	#elif defined HAS_FPGA

		#define INSTALL_COMPUTE_FN(closure_name, basic_fn, mkl_fn, cuda_fn, fpga_fn, mpi_fn) \
		  template<> class FnBundle<closure_name> {\
		   public:\
			static void Call(const DataList& i, const DataList& o, closure_name& c, const Context& context) {\
			  switch (context.impl_type) {\
				case ImplType::kBasic: basic_fn(i, o, c); break;\
				case ImplType::kMkl: mkl_fn(i, o, c); break;\
				case ImplType::kFpga: fpga_fn(i, o, c, context); break; \
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

		#define INSTALL_DATAGEN_FN(closure_name, basic_fn, mkl_fn, cuda_fn, fpga_fn, mpi_fn) \
		  template<> class FnBundle<closure_name> {\
		   public:\
			static void Call(const DataList& d, closure_name& c, const Context& context) {\
			  switch (context.impl_type) {\
				case ImplType::kBasic: basic_fn(d, c); break;\
				case ImplType::kMkl: mkl_fn(d, c); break;\
				case ImplType::kFpga: fpga_fn(d, c, context); break; \
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

	#else

		#define INSTALL_COMPUTE_FN(closure_name, basic_fn, mkl_fn, cuda_fn, fpga_fn, mpi_fn) \
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
/*			static int GetSerializedSize(){ \
				return closure_name.GetSerializedSize(); \
			} */ \
		  };

		#define INSTALL_DATAGEN_FN(closure_name, basic_fn, mkl_fn, cuda_fn, fpga_fn, mpi_fn) \
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

	#endif //CUDA / FPGA CHECK


#else // HAS_MPI


	#if defined HAS_CUDA && defined HAS_FPGA

		#define INSTALL_COMPUTE_FN(closure_name, basic_fn, mkl_fn, cuda_fn, fpga_fn, mpi_fn) \
		  template<> class FnBundle<closure_name> {\
		   public:\
			static void Call(const DataList& i, const DataList& o, closure_name& c, const Context& context) {\
			  switch (context.impl_type) {\
				case ImplType::kBasic: basic_fn(i, o, c); break;\
				case ImplType::kMkl: mkl_fn(i, o, c); break;\
				case ImplType::kCuda: cuda_fn(i, o, c, context); break; \
				case ImplType::kFpga: fpga_fn(i, o, c, context); break; \
				default: NO_IMPL(i, o, c, context); break;\
			  }\
			}\
		  };

		#define INSTALL_DATAGEN_FN(closure_name, basic_fn, mkl_fn, cuda_fn, fpga_fn, mpi_fn) \
		  template<> class FnBundle<closure_name> {\
		   public:\
			static void Call(const DataList& d, closure_name& c, const Context& context) {\
			  switch (context.impl_type) {\
				case ImplType::kBasic: basic_fn(d, c); break;\
				case ImplType::kMkl: mkl_fn(d, c); break;\
				case ImplType::kCuda: cuda_fn(d, c, context); break;\
				case ImplType::kFpga: fpga_fn(d, c, context); break; \
				default: NO_IMPL(d, c, context); break;\
			  }\
			}\
		  };

	#elif defined HAS_CUDA

		#define INSTALL_COMPUTE_FN(closure_name, basic_fn, mkl_fn, cuda_fn, fpga_fn, mpi_fn) \
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

		#define INSTALL_DATAGEN_FN(closure_name, basic_fn, mkl_fn, cuda_fn, fpga_fn, mpi_fn) \
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

	#elif defined HAS_FPGA

		#define INSTALL_COMPUTE_FN(closure_name, basic_fn, mkl_fn, cuda_fn, fpga_fn, mpi_fn) \
		  template<> class FnBundle<closure_name> {\
		   public:\
			static void Call(const DataList& i, const DataList& o, closure_name& c, const Context& context) {\
			  switch (context.impl_type) {\
				case ImplType::kBasic: basic_fn(i, o, c); break;\
				case ImplType::kMkl: mkl_fn(i, o, c); break;\
				case ImplType::kFpga: fpga_fn(i, o, c, context); break; \
				default: NO_IMPL(i, o, c, context); break;\
			  }\
			}\
		  };

		#define INSTALL_DATAGEN_FN(closure_name, basic_fn, mkl_fn, cuda_fn, fpga_fn, mpi_fn) \
		  template<> class FnBundle<closure_name> {\
		   public:\
			static void Call(const DataList& d, closure_name& c, const Context& context) {\
			  switch (context.impl_type) {\
				case ImplType::kBasic: basic_fn(d, c); break;\
				case ImplType::kMkl: mkl_fn(d, c); break;\
				case ImplType::kFpga: fpga_fn(d, c, context); break; \
				default: NO_IMPL(d, c, context); break;\
			  }\
			}\
		  };

	#else

		#define INSTALL_COMPUTE_FN(closure_name, basic_fn, mkl_fn, cuda_fn, fpga_fn, mpi_fn) \
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

		#define INSTALL_DATAGEN_FN(closure_name, basic_fn, mkl_fn, cuda_fn, fpga_fn, mpi_fn) \
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

	#endif

#endif//end MPI / non-MPI

}
