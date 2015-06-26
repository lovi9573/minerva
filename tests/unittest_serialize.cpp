#include "unittest_main.h"
#include <cmath>
#include "op/physical_op.h"
#include "op/closure.h"

using namespace std;
using namespace minerva;

TEST(Serialize, Fillop) {
  //Setup
  FillOp op = FillOp();
  op.closure.val = 3.2f;

  //Serialize
  int size = op.GetSerializedSize();
  char buffer[size];
  op.Serialize(buffer);

  //Deserialize
  int offset = 0;
  int bytesconsumed = 0;
  int closuretype;
  DESERIALIZE(buffer, offset, closuretype, int)
  std::shared_ptr<ComputeFn> deop = FillOp::DeSerialize(buffer+offset,&bytesconsumed);
  offset += bytesconsumed;

  //Validate
  EXPECT_FLOAT_EQ(op.closure.val, std::dynamic_pointer_cast<FillOp>(deop)->closure.val);
  EXPECT_EQ(size, offset);
  EXPECT_EQ(FILLCLOSURE, closuretype);
}

TEST(Serialize, Reductionop) {
  //Setup
  ReductionOp op = ReductionOp();
  op.closure.type = ReductionType::kSum;
  op.closure.dims_to_reduce = Scale{1,2,3,4};

  //Serialize
  int size = op.GetSerializedSize();
  char buffer[size];
  int usedsize = op.Serialize(buffer);
  EXPECT_EQ(size, usedsize);

  //Deserialize
  int offset = 0;
  int bytesconsumed = 0;
  int closuretype;
  DESERIALIZE(buffer, offset, closuretype, int)
  EXPECT_EQ(REDUCTIONCLOSURE, closuretype);
  std::shared_ptr<ComputeFn> deop = ReductionOp::DeSerialize(buffer+offset,&bytesconsumed);
  offset += bytesconsumed;

  //Validate
  EXPECT_EQ(size, offset);
  EXPECT_EQ(op.closure.type, std::dynamic_pointer_cast<ReductionOp>(deop)->closure.type);
  EXPECT_EQ(op.closure.dims_to_reduce, std::dynamic_pointer_cast<ReductionOp>(deop)->closure.dims_to_reduce);
}

TEST(Serialize, ArrayLoaderOp) {
  //Setup
  float array[] = {1.1,2.2,3.3,4.4,5.5,6.6};
  ArrayLoaderOp op = ArrayLoaderOp();
  op.closure.count = 6;
  op.closure.data = std::shared_ptr<float>(array, [](float* p) {});

  //Serialize
  int size = op.GetSerializedSize();
  char buffer[size];
  int usedsize = op.Serialize(buffer);
  EXPECT_EQ(size, usedsize);

  //Deserialize
  int offset = 0;
  int bytesconsumed = 0;
  int closuretype;
  DESERIALIZE(buffer, offset, closuretype, int)
  EXPECT_EQ(ARRAYLOADERCLOSURE, closuretype);
  std::shared_ptr<ComputeFn> deop = ArrayLoaderOp::DeSerialize(buffer+offset,&bytesconsumed);
  offset += bytesconsumed;

  //Validate
  EXPECT_EQ(size, offset);
  EXPECT_EQ(op.closure.count, std::dynamic_pointer_cast<ArrayLoaderOp>(deop)->closure.count);
  for(int i = 0; i < op.closure.count; i++){
	  EXPECT_FLOAT_EQ(*(op.closure.data.get()+i), *(std::dynamic_pointer_cast<ArrayLoaderOp>(deop)->closure.data.get()+i));
  }
}


TEST(Serialize, PhysicalOpFillOp) {
  //Setup
  FillOp *fop = new FillOp();
  fop->closure.val = 3.2f;
  uint64_t d_id = 12;
  PhysicalOp op = PhysicalOp(std::shared_ptr<FillOp>(fop), d_id);

  //Serialize
  int size = op.GetSerializedSize();
  char buffer[size];
  op.Serialize(buffer);

  //Deserialize
  int bytesconsumed = 0;
  PhysicalOp opout = PhysicalOp::DeSerialize(buffer, &bytesconsumed);

  //Validate
  EXPECT_EQ(d_id, opout.device_id);
  EXPECT_FLOAT_EQ(fop->closure.val, std::dynamic_pointer_cast<FillOp>(opout.compute_fn)->closure.val);
  EXPECT_EQ(size, bytesconsumed);
}

TEST(Serialize, Scale) {
  //Setup
  Scale sc = Scale({1,2,3,4});

  //Serialize
  int size = sc.GetSerializedSize();
  char buffer[size];
  sc.Serialize(buffer);

  //Deserialize
  int bytesconsumed = 0;
  Scale scout = Scale::DeSerialize(buffer, &bytesconsumed);

  //Validate
  EXPECT_EQ(Scale({1,2,3,4}), scout);
  EXPECT_EQ(size, bytesconsumed);
}

TEST(Serialize, PhysicalData) {
  //Setup
  uint64_t device_id =12;
  uint64_t data_id = 2;
  int rank = 4;
  PhysicalData pd = PhysicalData(Scale({1,2,3,4}),rank,device_id,data_id);
  //pd.rank = rank;

  //Serialize
  int size = pd.GetSerializedSize();
  char buffer[size];
  pd.Serialize(buffer);

  //Deserialize
  int bytesconsumed = 0;
  PhysicalData pdout = PhysicalData::DeSerialize(buffer, &bytesconsumed);

  //Validate
  EXPECT_EQ(pd.device_id, pdout.device_id);
  EXPECT_EQ(pd.data_id, pdout.data_id);
  EXPECT_EQ(pd.rank, pdout.rank);
  EXPECT_EQ(pd.size, pdout.size);
  EXPECT_EQ(size, bytesconsumed);
}


TEST(Serialize, TaskFill) {
  //Setup
  Task task = Task();
  int n = 4;
  for(int i = 0; i < n; i++){
	  task.inputs.emplace_back(TaskData(PhysicalData(Scale{1,2,3,4}, i, 2*i),1));
	  task.outputs.emplace_back(TaskData(PhysicalData(Scale{4,3,2,1}, 3*i, 4*i),1));
  }
  FillOp *fop = new FillOp();
  fop->closure.val = 3.2f;
  uint64_t d_id = 12;
  task.op = PhysicalOp(std::shared_ptr<FillOp>(fop), d_id);
  task.id =12;
  //printf("Setup Done\n");
  //fflush(stdout);

  //Serialize
  int size = task.GetSerializedSize();
  char buffer[size];
  task.Serialize(buffer);
  //printf("Serialize Done\n");
  //fflush(stdout);

  //Deserialize
  int bytesconsumed = 0;
  Task taskout = Task::DeSerialize(buffer, &bytesconsumed);
  //printf("DeSerialize Done\n");
  //fflush(stdout);

  //Validate
  EXPECT_EQ(size, bytesconsumed);
  EXPECT_EQ(task.id, taskout.id);
  EXPECT_EQ(task.op.device_id, taskout.op.device_id);
  EXPECT_EQ(std::dynamic_pointer_cast<FillOp>(task.op.compute_fn)->closure.val, std::dynamic_pointer_cast<FillOp>(taskout.op.compute_fn)->closure.val);
  for(int i = 0; i < n; i++){
  	    EXPECT_EQ(task.inputs.at(i).physical_data.size, taskout.inputs.at(i).physical_data.size);
		EXPECT_EQ(task.inputs.at(i).physical_data.rank, taskout.inputs.at(i).physical_data.rank);
		EXPECT_EQ(task.inputs.at(i).physical_data.data_id, taskout.inputs.at(i).physical_data.data_id);
		EXPECT_EQ(task.inputs.at(i).physical_data.device_id, taskout.inputs.at(i).physical_data.device_id);
  	    EXPECT_EQ(task.outputs.at(i).physical_data.size, taskout.outputs.at(i).physical_data.size);
		EXPECT_EQ(task.outputs.at(i).physical_data.rank, taskout.outputs.at(i).physical_data.rank);
		EXPECT_EQ(task.outputs.at(i).physical_data.data_id, taskout.outputs.at(i).physical_data.data_id);
		EXPECT_EQ(task.outputs.at(i).physical_data.device_id, taskout.outputs.at(i).physical_data.device_id);
    }
}

TEST(Serialize, TaskAdd) {
  //Setup
  Task task = Task();
  int n = 2;
  for(int i = 0; i < n; i++){
	  task.inputs.emplace_back(TaskData(PhysicalData(Scale{1,2,3,4}, i, 2*i),1));
  }
  task.outputs.emplace_back(TaskData(PhysicalData(Scale{1,2,3,4}, 10, 11),1));
  ArithmeticOp* arith_op = new ArithmeticOp();
  arith_op->closure = {ArithmeticType::kAdd};
  int device_id = 4;
  task.op = PhysicalOp(std::shared_ptr<ComputeFn>(arith_op), device_id );
  task.id =12;
  //printf("Setup Done\n");
  //fflush(stdout);

  //Serialize
  int size = task.GetSerializedSize();
  char buffer[size];
  task.Serialize(buffer);
  //printf("Serialize Done\n");
  //fflush(stdout);

  //Deserialize
  int bytesconsumed = 0;
  Task taskout = Task::DeSerialize(buffer, &bytesconsumed);
  //printf("DeSerialize Done\n");
  //fflush(stdout);

  //Validate
  EXPECT_EQ(size, bytesconsumed);
  EXPECT_EQ(task.id, taskout.id);
  EXPECT_EQ(task.op.device_id, taskout.op.device_id);
  EXPECT_EQ(std::dynamic_pointer_cast<ArithmeticOp>(task.op.compute_fn)->closure.type, std::dynamic_pointer_cast<ArithmeticOp>(taskout.op.compute_fn)->closure.type);
  for(int i = 0; i < n; i++){
  	    EXPECT_EQ(task.inputs.at(i).physical_data.size, taskout.inputs.at(i).physical_data.size);
		EXPECT_EQ(task.inputs.at(i).physical_data.rank, taskout.inputs.at(i).physical_data.rank);
		EXPECT_EQ(task.inputs.at(i).physical_data.data_id, taskout.inputs.at(i).physical_data.data_id);
		EXPECT_EQ(task.inputs.at(i).physical_data.device_id, taskout.inputs.at(i).physical_data.device_id);
    }
	EXPECT_EQ(task.outputs.at(0).physical_data.size, taskout.outputs.at(0).physical_data.size);
	EXPECT_EQ(task.outputs.at(0).physical_data.rank, taskout.outputs.at(0).physical_data.rank);
	EXPECT_EQ(task.outputs.at(0).physical_data.data_id, taskout.outputs.at(0).physical_data.data_id);
	EXPECT_EQ(task.outputs.at(0).physical_data.device_id, taskout.outputs.at(0).physical_data.device_id);
}
/*
//TODO: Complete this
TEST(Serialize, TaskSum) {
  //Setup
  Task task = Task();
  int n = 2;
  for(int i = 0; i < n; i++){
	  task.inputs.emplace_back(TaskData(PhysicalData(Scale{1,2,3,4}, i, 2*i),1));
  }
  task.outputs.emplace_back(TaskData(PhysicalData(Scale{1,2,3,4}, 10, 11),1));
  ArithmeticOp* arith_op = new ArithmeticOp();
  arith_op->closure = {ArithmeticType::kAdd};
  int device_id = 4;
  task.op = PhysicalOp(std::shared_ptr<ComputeFn>(arith_op), device_id );
  task.id =12;
  //printf("Setup Done\n");
  //fflush(stdout);

  //Serialize
  int size = task.GetSerializedSize();
  char buffer[size];
  task.Serialize(buffer);
  //printf("Serialize Done\n");
  //fflush(stdout);

  //Deserialize
  int bytesconsumed = 0;
  Task taskout = Task::DeSerialize(buffer, &bytesconsumed);
  //printf("DeSerialize Done\n");
  //fflush(stdout);

  //Validate
  EXPECT_EQ(size, bytesconsumed);
  EXPECT_EQ(task.id, taskout.id);
  EXPECT_EQ(task.op.device_id, taskout.op.device_id);
  EXPECT_EQ(std::dynamic_pointer_cast<ArithmeticOp>(task.op.compute_fn)->closure.type, std::dynamic_pointer_cast<ArithmeticOp>(taskout.op.compute_fn)->closure.type);
  for(int i = 0; i < n; i++){
  	    EXPECT_EQ(task.inputs.at(i).physical_data.size, taskout.inputs.at(i).physical_data.size);
		EXPECT_EQ(task.inputs.at(i).physical_data.rank, taskout.inputs.at(i).physical_data.rank);
		EXPECT_EQ(task.inputs.at(i).physical_data.data_id, taskout.inputs.at(i).physical_data.data_id);
		EXPECT_EQ(task.inputs.at(i).physical_data.device_id, taskout.inputs.at(i).physical_data.device_id);
    }
	EXPECT_EQ(task.outputs.at(0).physical_data.size, taskout.outputs.at(0).physical_data.size);
	EXPECT_EQ(task.outputs.at(0).physical_data.rank, taskout.outputs.at(0).physical_data.rank);
	EXPECT_EQ(task.outputs.at(0).physical_data.data_id, taskout.outputs.at(0).physical_data.data_id);
	EXPECT_EQ(task.outputs.at(0).physical_data.device_id, taskout.outputs.at(0).physical_data.device_id);
}*/
