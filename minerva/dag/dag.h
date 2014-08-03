#pragma once

#include <cstdint>
#include <vector>
#include <map>
#include <set>
#include <functional>
#include <initializer_list>
#include <atomic>
#include <string>

#include "common/common.h"
#include "common/concurrent_blocking_queue.h"

namespace minerva {

class DagNode {
 public:
  enum NodeTypes {
    OP_NODE = 0,
    DATA_NODE
  };
  virtual ~DagNode() {}
  void AddParent(DagNode*);
  void AddParents(std::initializer_list<DagNode*>);
  bool DeleteParent(DagNode*);
  virtual NodeTypes Type() const = 0;

  uint64_t node_id() const { return node_id_; }
  void set_node_id(uint64_t id) { node_id_ = id; }

  // TODO Use unordered version for quicker access
  std::set<DagNode*> successors_;
  std::set<DagNode*> predecessors_;
 private:
  uint64_t node_id_;
};

template<typename Data, typename Op>
class DataNode : public DagNode {
 public:
  DataNode() {}
  ~DataNode();
  NodeTypes Type() const { return DagNode::DATA_NODE; }
  Data data_;

 private:
  DISALLOW_COPY_AND_ASSIGN(DataNode);
};

template<typename Data, typename Op>
class OpNode : public DagNode {
  typedef DataNode<Data, Op> DNode;
 public:
  OpNode() {}
  ~OpNode();
  NodeTypes Type() const { return DagNode::OP_NODE; }
  Op op_;
  std::vector<DNode*> inputs_, outputs_;

 private:
  DISALLOW_COPY_AND_ASSIGN(OpNode);
};

template<typename Data, typename Op>
class DagHelper {
 public:
  static std::string DataToString(const Data& d) {
    return "N/A";
  }
  static std::string OpToString(const Op& o) {
    return "N/A";
  }
  static void FreeData(Data& d) {}
  static void FreeOp(Op& o) {}
};

template<class DagType>
class DagMonitor {
 public:
  virtual void OnCreateNode(DagNode* );
  virtual void OnDeleteNode(DagNode* );
  virtual void OnCreateDataNode(typename DagType::DNode* ) {}
  virtual void OnCreateOpNode(typename DagType::ONode* ) {}
  virtual void OnDeleteDataNode(typename DagType::DNode* ) {}
  virtual void OnDeleteOpNode(typename DagType::ONode* ) {}
};

template<class Data, class Op>
class Dag {
 public:
  typedef DataNode<Data, Op> DNode;
  typedef OpNode<Data, Op> ONode;
  Dag() {}
  ~Dag();
  DNode* NewDataNode(const Data& data);
  ONode* NewOpNode(std::vector<DNode*> inputs,
      std::vector<DNode*> outputs, const Op& op);
  void DeleteNode(uint64_t );
  bool ExistNode(uint64_t ) const;
  DagNode* GetNode(uint64_t ) const;
  ONode* GetOpNode(uint64_t ) const;
  DNode* GetDataNode(uint64_t ) const;

  // TODO use unordered_map instead
  typedef std::map<uint64_t, DagNode*> ContainerType;
  ContainerType::iterator begin() { return index_to_node_.begin(); }
  ContainerType::iterator end() { return index_to_node_.end(); }

  void RegisterMonitor(DagMonitor<Dag<Data, Op>>* );
  template<class NodePrinter=DagHelper<Data, Op> >
  std::string PrintDag() const;

 private:
  DISALLOW_COPY_AND_ASSIGN(Dag);
  uint64_t NewIndex();

 private:
  std::vector<DagMonitor<Dag<Data, Op>>*> monitors_;
  ContainerType index_to_node_;
};

} // end of namespace minerva

#include "dag.inl"
