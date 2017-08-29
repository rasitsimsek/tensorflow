/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

//#include <cstdio>
//#include <functional>
//#include <string>
//#include <vector>
//
//#include "tensorflow/cc/ops/standard_ops.h"
//#include "tensorflow/core/framework/graph.pb.h"
//#include "tensorflow/core/framework/tensor.h"
//#include "tensorflow/core/graph/default_device.h"
//#include "tensorflow/core/graph/graph_def_builder.h"
//#include "tensorflow/core/lib/core/threadpool.h"
//#include "tensorflow/core/lib/strings/stringprintf.h"
//#include "tensorflow/core/platform/init_main.h"
//#include "tensorflow/core/platform/logging.h"
//#include "tensorflow/core/platform/types.h"
//#include "tensorflow/core/public/session.h"
//
//using tensorflow::string;
//using tensorflow::int32;
//
//namespace tensorflow {
//namespace example {
//
//struct Options {
//  int num_concurrent_sessions = 1;   // The number of concurrent sessions
//  int num_concurrent_steps = 10;     // The number of concurrent steps
//  int num_iterations = 100;          // Each step repeats this many times
//  bool use_gpu = false;              // Whether to use gpu in the training
//};
//
//// A = [3 2; -1 0]; x = rand(2, 1);
//// We want to compute the largest eigenvalue for A.
//// repeat x = y / y.norm(); y = A * x; end
//GraphDef CreateGraphDef() {
//  // TODO(jeff,opensource): This should really be a more interesting
//  // computation.  Maybe turn this into an mnist model instead?
//  Scope root = Scope::NewRootScope();
//  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
//
//  // A = [3 2; -1 0].  Using Const<float> means the result will be a
//  // float tensor even though the initializer has integers.
//  auto a = Const<float>(root, {{3, 2}, {-1, 0}});
//
//  // x = [1.0; 1.0]
//  auto x = Const(root.WithOpName("x"), {{1.f}, {1.f}});
//
//  // y = A * x
//  auto y = MatMul(root.WithOpName("y"), a, x);
//
//  // y2 = y.^2
//  auto y2 = Square(root, y);
//
//  // y2_sum = sum(y2).  Note that you can pass constants directly as
//  // inputs.  Sum() will automatically create a Const node to hold the
//  // 0 value.
//  auto y2_sum = Sum(root, y2, 0);
//
//  // y_norm = sqrt(y2_sum)
//  auto y_norm = Sqrt(root, y2_sum);
//
//  // y_normalized = y ./ y_norm
//  Div(root.WithOpName("y_normalized"), y, y_norm);
//
//  GraphDef def;
//  TF_CHECK_OK(root.ToGraphDef(&def));
//
//  return def;
//}
//
//string DebugString(const Tensor& x, const Tensor& y) {
//  CHECK_EQ(x.NumElements(), 2);
//  CHECK_EQ(y.NumElements(), 2);
//  auto x_flat = x.flat<float>();
//  auto y_flat = y.flat<float>();
//  // Compute an estimate of the eigenvalue via
//  //      (x' A x) / (x' x) = (x' y) / (x' x)
//  // and exploit the fact that x' x = 1 by assumption
//  Eigen::Tensor<float, 0, Eigen::RowMajor> lambda = (x_flat * y_flat).sum();
//  return strings::Printf("lambda = %8.6f x = [%8.6f %8.6f] y = [%8.6f %8.6f]",
//                         lambda(), x_flat(0), x_flat(1), y_flat(0), y_flat(1));
//}
//
//void ConcurrentSteps(const Options* opts, int session_index) {
//  // Creates a session.
//  SessionOptions options;
//  std::unique_ptr<Session> session(NewSession(options));
//  GraphDef def = CreateGraphDef();
//  if (options.target.empty()) {
//    graph::SetDefaultDevice(opts->use_gpu ? "/gpu:0" : "/cpu:0", &def);
//  }
//
//  TF_CHECK_OK(session->Create(def));
//
//  // Spawn M threads for M concurrent steps.
//  const int M = opts->num_concurrent_steps;
//  std::unique_ptr<thread::ThreadPool> step_threads(
//      new thread::ThreadPool(Env::Default(), "trainer", M));
//
//  for (int step = 0; step < M; ++step) {
//    step_threads->Schedule([&session, opts, session_index, step]() {
//      // Randomly initialize the input.
//      Tensor x(DT_FLOAT, TensorShape({2, 1}));
//      auto x_flat = x.flat<float>();
//      x_flat.setRandom();
//      Eigen::Tensor<float, 0, Eigen::RowMajor> inv_norm =
//          x_flat.square().sum().sqrt().inverse();
//      x_flat = x_flat * inv_norm();
//
//      // Iterations.
//      std::vector<Tensor> outputs;
//      for (int iter = 0; iter < opts->num_iterations; ++iter) {
//        outputs.clear();
//        TF_CHECK_OK(
//            session->Run({{"x", x}}, {"y:0", "y_normalized:0"}, {}, &outputs));
//        CHECK_EQ(size_t{2}, outputs.size());
//
//        const Tensor& y = outputs[0];
//        const Tensor& y_norm = outputs[1];
//        // Print out lambda, x, and y.
//        std::printf("%06d/%06d %s\n", session_index, step,
//                    DebugString(x, y).c_str());
//        // Copies y_normalized to x.
//        x = y_norm;
//      }
//    });
//  }
//
//  // Delete the threadpool, thus waiting for all threads to complete.
//  step_threads.reset(nullptr);
//  TF_CHECK_OK(session->Close());
//}
//
//void ConcurrentSessions(const Options& opts) {
//  // Spawn N threads for N concurrent sessions.
//  const int N = opts.num_concurrent_sessions;
//
//  // At the moment our Session implementation only allows
//  // one concurrently computing Session on GPU.
//  CHECK_EQ(1, N) << "Currently can only have one concurrent session.";
//
//  thread::ThreadPool session_threads(Env::Default(), "trainer", N);
//  for (int i = 0; i < N; ++i) {
//    session_threads.Schedule(std::bind(&ConcurrentSteps, &opts, i));
//  }
//}
//
//}  // end namespace example
//}  // end namespace tensorflow
//
//namespace {
//
//bool ParseInt32Flag(tensorflow::StringPiece arg, tensorflow::StringPiece flag,
//                    int32* dst) {
//  if (arg.Consume(flag) && arg.Consume("=")) {
//    char extra;
//    return (sscanf(arg.data(), "%d%c", dst, &extra) == 1);
//  }
//
//  return false;
//}
//
//bool ParseBoolFlag(tensorflow::StringPiece arg, tensorflow::StringPiece flag,
//                   bool* dst) {
//  if (arg.Consume(flag)) {
//    if (arg.empty()) {
//      *dst = true;
//      return true;
//    }
//
//    if (arg == "=true") {
//      *dst = true;
//      return true;
//    } else if (arg == "=false") {
//      *dst = false;
//      return true;
//    }
//  }
//
//  return false;
//}
//
//}  // namespace
//
//int main(int argc, char* argv[]) {
//  tensorflow::example::Options opts;
//  std::vector<char*> unknown_flags;
//  for (int i = 1; i < argc; ++i) {
//    if (string(argv[i]) == "--") {
//      while (i < argc) {
//        unknown_flags.push_back(argv[i]);
//        ++i;
//      }
//      break;
//    }
//
//    if (ParseInt32Flag(argv[i], "--num_concurrent_sessions",
//                       &opts.num_concurrent_sessions) ||
//        ParseInt32Flag(argv[i], "--num_concurrent_steps",
//                       &opts.num_concurrent_steps) ||
//        ParseInt32Flag(argv[i], "--num_iterations", &opts.num_iterations) ||
//        ParseBoolFlag(argv[i], "--use_gpu", &opts.use_gpu)) {
//      continue;
//    }
//
//    fprintf(stderr, "Unknown flag: %s\n", argv[i]);
//    return -1;
//  }
//
//  // Passthrough any unknown flags.
//  int dst = 1;  // Skip argv[0]
//  for (char* f : unknown_flags) {
//    argv[dst++] = f;
//  }
//  argv[dst++] = nullptr;
//  argc = static_cast<int>(unknown_flags.size() + 1);
//  tensorflow::port::InitMain(argv[0], &argc, &argv);
//  tensorflow::example::ConcurrentSessions(opts);
//}


#include <iostream>
#include <vector>
#include <string>
#include <unordered_set>

//#include <QApplication>
//#include "MainWindow.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/edit_distance.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"
#include "tensorflow/core/protobuf/config.pb.h"

using namespace std;

void makeIndex( std::vector<string>&& rsoStringVector, tensorflow::Tensor& roIndexTensor )
{
   auto ix_t = roIndexTensor.matrix<int64_t>();
   std::size_t stCounter = 0;
   for( std::size_t stX = 0; stX < rsoStringVector.size(); stX++ )
   {
      const std::string& rsString = rsoStringVector[ stX ];
      for( std::size_t stY = 0; stY < rsString.size(); stY++ )
      {
         ix_t( stCounter, 0 ) = stX;
         ix_t( stCounter, 1 ) = 0;
         ix_t( stCounter, 2 ) = stY;
         stCounter++;
      }
   }
}

void makeIndex( const std::vector<string>& rsoStringVector, tensorflow::Tensor& roIndexTensor )
{
   auto ix_t = roIndexTensor.matrix<int64_t>();
   std::size_t stCounter = 0;
   for( std::size_t stX = 0; stX < rsoStringVector.size(); stX++ )
   {
      const std::string& rsString = rsoStringVector[ stX ];
      for( std::size_t stY = 0; stY < rsString.size(); stY++ )
      {
         ix_t( stCounter, 0 ) = stX;
         ix_t( stCounter, 1 ) = 0;
         ix_t( stCounter, 2 ) = stY;
         stCounter++;
      }
   }
}

void makeChar( std::vector<std::string>&& rsoStringVector, tensorflow::Tensor& roCharTensor )
{
   auto vals_t = roCharTensor.vec<string>();

   int64_t i64Index = 0;
   for( std::size_t stX = 0; stX < rsoStringVector.size(); stX++ )
   {
      const std::string& rsString = rsoStringVector[ stX ];
      for( std::size_t stY = 0; stY < rsString.size(); stY++ )
      {
         vals_t( i64Index++ ) = rsString[ stY ];
      }
   }
}

void makeChar( const std::vector<std::string>& rsoStringVector, tensorflow::Tensor& roCharTensor )
{
   auto vals_t = roCharTensor.vec<string>();

   int64_t i64Index = 0;
   for( std::size_t stX = 0; stX < rsoStringVector.size(); stX++ )
   {
      const std::string& rsString = rsoStringVector[ stX ];
      for( std::size_t stY = 0; stY < rsString.size(); stY++ )
      {
         vals_t( i64Index++ ) = rsString[ stY ];
      }
   }
}

void makeFloat( const std::vector<int>& roNumVector, tensorflow::Tensor& roNumTensor )
{
   auto vals_t = roNumTensor.matrix<float>();

   int64_t i64Index = 0;
   for( std::size_t stX = 0; stX < roNumVector.size(); stX++ )
   {
      const auto roVal = roNumVector[ stX ];
      vals_t( 0, i64Index ) = roVal;
      i64Index++;
   }
}

void testEditDistanse()
{
   using namespace tensorflow;
   using namespace tensorflow::ops;
   using namespace tensorflow::gtl;

   Scope root = Scope::NewRootScope();

   vector<string> street_names = { "röntgen", "mannheimer", "donner", "donnersberg", "haupt" };
   vector<string> street_types = { "str", "weg", "gasse" };

   vector<int> rand_zips = { 65001, 65502, 65002, 64354, 65344 };
   // Reference data
   //vector<int> numbers = { 44, 44, 67, 156, 321, 545, 23, 45, 856, 2 };
   //vector<string> streets = { "röntgen", "donner", "haupt", "mannheimer", "donnersberg", "donner", "haupt" };
   //vector<string> streets_suffx = { "str"    , "weg"   , "str"  , "str"       , "str"        , "gasse" , "weg" };
   vector<int> numbers = { 44, 44 };
   vector<string> streets = { "röntgen", "römer" };
   vector<string> streets_suffx = { "str"    , "str."   , "str"  , "str"       , "str"        , "gasse" , "weg" };
   vector<int> zips = { 65344 , 55344 };

   vector<string> full_streets;
   vector<string> reference_data;

   for( size_t stIndex = 0; stIndex < streets.size() ; ++stIndex )
   {
      full_streets.emplace_back( streets[ stIndex ] + " " + streets_suffx[ stIndex ] + " " + to_string( numbers[ stIndex ] ) );
   }

   int64 stRefSize = 0;
   for( size_t stIndex = 0; stIndex < full_streets.size(); ++stIndex )
   {
      reference_data.emplace_back(  to_string( zips[ stIndex ] ) + " " + full_streets[ stIndex ] );
      stRefSize += reference_data.back().size();
   }

   const int64_t kstN = full_streets.size();
   Tensor ix( DT_INT64, TensorShape( { static_cast<int64_t>( stRefSize ), 3i64 } ) );
   Tensor vals( DT_STRING, TensorShape( { static_cast<int64_t>( stRefSize ) } ) );
   Tensor zip_refs( DT_FLOAT, TensorShape( { 1i64, kstN } ) );

   makeIndex( reference_data, ix );
   makeChar( reference_data, vals );
   makeFloat( zips, zip_refs );

   //LOG( INFO )<< "Indexes = " << ix.SummarizeValue( stRefSize );
   //sparse::SparseTensor oSparse_Refs_Set( ix, vals, TensorShape( { static_cast<int64_t>( reference_data.size() ), 1i64, 1i64 } ) );

   std::cout << "Street Size = " << full_streets.size() << "\n"
      << "Ref Size =    " << stRefSize << "\n";
   //LOG( INFO ) << oSparse_Refs_Set.indices().SummarizeValue(stRefSize);

   std::string sTestAddress = "55344 römer str. 44";
   vector<int> iTestZip = { 55344 };

   std::vector<std::string> sRepeatEntry( kstN, sTestAddress );

   const int64_t i64TestSize = sRepeatEntry[ 0 ].size() * kstN;

   Tensor ix_test( DT_INT64, TensorShape( { i64TestSize, 3i64 } ) );
   Tensor vals_test( DT_STRING, TensorShape( { i64TestSize } ) );
   Tensor zip_test( DT_FLOAT, TensorShape( { 1i64, 1i64 } ) );

   makeIndex( sRepeatEntry, ix_test );
   makeChar( sRepeatEntry, vals_test );
   makeFloat( iTestZip, zip_test );

   //sparse::SparseTensor oSparse_Test_Set( ix_test, vals_test, TensorShape( { i64TestSize, 1i64, 1i64 } ) );

   auto test_address = Placeholder( root.WithOpName( "test_address" ), DT_STRING );
   auto test_address_shape = Placeholder::Shape( PartialTensorShape{ i64TestSize, 3i64 } );
   auto test_address_ix = Placeholder( root.WithOpName( "test_address_ix" ), DT_INT64, test_address_shape );
   auto test_zip = Placeholder( root.WithOpName( "test_zip" ), DT_FLOAT, Placeholder::Shape( PartialTensorShape{ -1i64,1i64 } ) );
   auto ref_address = Placeholder( root.WithOpName( "ref_address" ), DT_STRING );
   auto ref_address_shape = Placeholder::Shape( PartialTensorShape{stRefSize, 3i64 } );
   auto ref_address_ix = Placeholder( root.WithOpName( "ref_address_ix" ), DT_INT64, ref_address_shape );
   auto ref_zip = Placeholder( root.WithOpName( "ref_zip" ), DT_FLOAT, Placeholder::Shape( PartialTensorShape{ -1i64, kstN } ) );

   std::cout << "Test Street Size = " << sRepeatEntry.size() << "\n"
      << "Test Size        = " << sRepeatEntry[ 0 ].size() * kstN << "\n"
      << "Test Vals =" << vals_test.SummarizeValue(100) << "\n"
      << "Index     =" << ix_test.SummarizeValue( 100 ) << "\n";


   // Declaration of zip distance calculation 
   auto zip_dist = Square( root, Subtract( root, ref_zip, test_zip ) );

   // Declaration of edit distance of address
   auto address_dist = EditDistance( root
                                     , test_address_ix
                                     , test_address
                                     , { kstN,1i64,1i64 }// {static_cast<int64_t>(1),3}//test_address_shape
                                     , ref_address_ix
                                     , ref_address
                                     , { kstN,1i64,1i64 } //ref_address_shape
   , EditDistance::Normalize( true ) );


   ArraySlice<int64> perm_vect = {1i64,0i64};



   auto zip_max = Gather( root, Squeeze( root, zip_dist ), ArgMax( root, zip_dist, 1i64 ) );
   auto zip_min = Gather( root, Squeeze( root, zip_dist ), ArgMin( root, zip_dist, 1i64 ) );
   auto zip_sim = Div( root, Subtract( root, zip_max, zip_dist ), Subtract( root, zip_max, zip_min ) );
   auto address_sim = Subtract( root, 1.0f, address_dist );
   const float address_weight = 0.5f;
   const float zip_weight = 1.0f - address_weight;
   auto weight_address = Multiply( root, address_weight, address_sim );
   auto transpose = Transpose( root, weight_address, perm_vect );
  // auto weighted_sim = Add( root, transpose, Multiply( root, zip_weight, zip_sim ) );

   // weighted_sim = tf.add(tf.transpose(tf.multiply(address_weight, address_sim)), tf.multiply(zip_weight, zip_sim))
 //  auto top_match_index = ArgMax( root.WithOpName( "top_match_index" ), weighted_sim, 1i64 );


   ClientSession::FeedType inputs = {
                                        {test_address_ix, ix_test}
                                      , {test_address   , vals_test }
                                      , {ref_address_ix , ix}
                                      , {ref_address    , vals }
                                      , {test_zip       , zip_test                  }
                                      , {ref_zip        , zip_refs                  }
                                      , { perm_ph       , permut    }
   };

//   const std::vector<Output> outputs = { address_dist };
   std::vector<Tensor> output_tensor;
   ClientSession session( root );
   std::cout << root.status().error_message();

   if( !root.status().ok() )
      return;

   RunOptions oRunOptions;
   oRunOptions.set_trace_level( RunOptions::FULL_TRACE );
   RunMetadata run_metadata;

   session.Run( inputs, { transpose }, &output_tensor );
   if( !output_tensor.empty() )
      std::cout << output_tensor[ 0 ].SummarizeValue( 300 );
}


void simpleTest()
{
   using namespace tensorflow;
   using namespace tensorflow::ops;
   using namespace tensorflow::gtl;


   Scope root = Scope::NewRootScope();

   Tensor hypho2_ix( DT_INT64, TensorShape( { static_cast<int64_t>( 5 ), 3LL } ) );
   Tensor hypho2_vals( DT_STRING, TensorShape( { static_cast<int64_t>( 5 ) } ) );
   //   hypho2_vals.flat<string>().setConstant("bear");
   makeIndex( { "beers" }, hypho2_ix );
   makeChar( { "beers" }, hypho2_vals );

   Tensor truth2_ix( DT_INT64, TensorShape( { static_cast<int64_t>( 10 ), 3LL } ) );
   Tensor truth2_vals( DT_STRING, TensorShape( { static_cast<int64_t>( 10 ) } ) );
   //truth2_vals.flat<string>().setConstant( "beers" );

   makeIndex( { "beers" }, truth2_ix );
   makeChar( { "beersbaers" }, truth2_vals );


   auto test_val             = Placeholder( root, DT_STRING );
   auto test_ix_shape        = Placeholder::Shape( PartialTensorShape{ -1LL, 3LL } );
   auto test_ix              = Placeholder( root, DT_INT64, test_ix_shape );
   //auto test_zip           = Placeholder( root.WithOpName("test_zip"), DT_FLOAT, Placeholder::Shape( PartialTensorShape( { -1,1 } ) ) );
   auto ref_val              = Placeholder( root, DT_STRING );
   auto ref_ix_shape         = Placeholder::Shape( PartialTensorShape{ -1LL, 3LL } );
   auto ref_ix               = Placeholder( root, DT_INT64, ref_ix_shape );
   //auto ref_zip            = Placeholder( root.WithOpName("ref_zip"), DT_FLOAT, Placeholder::Shape( PartialTensorShape( { -1, kstN } ) ) );


     //   auto c = Subtract( root, a, truth2_ix );
  //   auto address_sim = Subtract( root, test_address_ix, ref_address_ix );

     // Declaration of edit distance of address
     //auto address_dist = EditDistance( root.WithOpName("address:dist")
     //                        , test_address_ix
     //                        , test_address
     //                        , {1,1,1}// {static_cast<int64_t>(1),3}//test_address_shape
     //                        , ref_address_ix
     //                        , ref_address
     //                        , {1,1,1} //ref_address_shape
     //                        , EditDistance::Normalize(false) );

   auto address_dist = EditDistance( root
                                     , test_ix
                                     , test_val
                                     , { 1LL,1LL,1LL }// {static_cast<int64_t>(1),3}//test_address_shape
                                     , ref_ix
                                     , ref_val
                                     , { 2LL,1LL,1LL } //ref_address_shape
                                     , EditDistance::Normalize( true ) );

   auto address_sim = Subtract( root, 1.0f, address_dist );

   //   auto oOutput =  address_dist.output(nullptr);
   ClientSession session( root );
   std::vector<Tensor> outputs;

   RunOptions oRunOptions;
   oRunOptions.set_trace_level( RunOptions::FULL_TRACE );

   RunMetadata run_metadata;

   LOG( INFO ) << hypho2_ix.SummarizeValue( 200 ) << "\n";
   LOG( INFO ) << hypho2_vals.SummarizeValue( 200 ) << "\n";

   std::cout << root.status().error_message();

   if( !root.status().ok() )
      return;

   session.Run( /*oRunOptions,*/ 
               {
                  {test_ix, hypho2_ix},
                  { ref_ix, truth2_ix },
                  { test_val, hypho2_vals },
                  { ref_val, truth2_vals }
               },
               { address_sim }, /*{},*/
               &outputs/*, &run_metadata*/ );

   cout << outputs[ 0 ].SummarizeValue( 1 );
}

int main( int argc, char *argv[] )
{
//   simpleTest();
   testEditDistanse();
   return 0;

   //   QApplication oApp( argc, argv );
   //   MainWindow oMainWindow;
   //   oMainWindow.show();
   //   return oApp


   //Scope root = Scope::NewRootScope();
   //// Matrix A = [3 2; -1 0]
   //auto A = Const( root, { {3.f, 2.f}, {-1.f, 0.f} } );
   //// Vector b = [3 5]
   //auto b = Const( root, { {3.f, 5.f} } );
   //// v = Ab^T
   //auto v = MatMul( root.WithOpName( "v" ), A, b, MatMul::TransposeB( true ) );
   //std::vector<Tensor> output_tensor;
   //ClientSession session( root );
   //// Run and fetch v
   //TF_CHECK_OK( session.Run( { v }, &outputs ) );
   //// Expect outputs[0] == [19; -3]
   //LOG( INFO ) << outputs[ 0 ].matrix<float>();


   //vector<string> street_names = { "röntgen", "mannheimer", "donner", "donnersberg", "haupt" };
   //vector<string> street_types = { "str", "weg", "gasse" };

   //vector<int> rand_zips = {65001, 65502, 65002, 64354, 65344 };
   //// Reference data
   //vector<int> numbers = { 19, 44, 67, 156, 321, 545, 23, 45, 856, 2 };
   //vector<string> streets       = { "röntgen", "donner", "haupt", "mannheimer", "donnersberg", "donner", "haupt" };
   //vector<string> streets_suffx = { "str"    , "weg"   , "str"  , "str"       , "str"        , "gasse" , "weg"   };
   //vector<int> zips             = {65001     , 65502   , 65002  , 64354       , 65344        , 65244   , 67234   };

   //vector<string> full_streets;
   //vector<string> reference_data;

   //for( size_t stIndex = 0 ; stIndex < streets.size() ; ++stIndex )
   //{
   //   full_streets.emplace_back( streets[stIndex] + " " + streets_suffx[stIndex] + " " + to_string( numbers[stIndex] ) );
   //}

   //int64 stRefSize = 0;
   //for( size_t stIndex = 0; stIndex < full_streets.size(); ++stIndex )
   //{
   //   reference_data = { full_streets[stIndex] + " " +  to_string(zips[stIndex]) };
   //   stRefSize += reference_data.back().size();
   //}

   //const int64_t kstN = full_streets.size();
   //Tensor ix(DT_INT64, TensorShape({static_cast<int64_t>( stRefSize ), 3}));
   //Tensor vals(DT_STRING, TensorShape({static_cast<int64_t>( stRefSize )}));
   //Tensor zip_refs( DT_FLOAT, TensorShape( { static_cast<int64_t>( kstN ) } ) );

   //makeIndex( full_streets, ix, stRefSize );
   //makeChar( full_streets, vals );
   //makeFloat(zips,  zip_refs );
   //
   //LOG( INFO )<< "Indexes = " << ix.SummarizeValue( stRefSize );
   //sparse::SparseTensor oSparse_Refs_Set( ix, vals, TensorShape( { static_cast<int64_t>( reference_data.size() ), 1, 1 } ) );

   //std::cout << "Street Size = " << full_streets.size() << "\n"
   //          << "Ref Size =    " << stRefSize << "\n";
   //LOG( INFO ) << oSparse_Refs_Set.indices().SummarizeValue(stRefSize);




   //std::string sTestAddress = "65344 röstgenstr 44";
   //vector<int> iTestZip = { 65344 };

   //std::vector<std::string> sRepeatEntry( kstN, sTestAddress );

   //const int64_t i64TestSize = sRepeatEntry[ 0 ].size() * kstN;

   //Tensor ix_test( DT_INT64, TensorShape( { i64TestSize, 3 } ) );
   //Tensor vals_test( DT_STRING, TensorShape( { i64TestSize } ) );
   //Tensor zip_test( DT_FLOAT, TensorShape( { static_cast<int64_t>( 1 ) } ) );

   //makeIndex( sRepeatEntry, ix_test, sRepeatEntry[ 0 ].size() * kstN );
   //makeChar( sRepeatEntry , vals_test );
   //makeFloat( iTestZip, zip_test );

   //sparse::SparseTensor oSparse_Test_Set( ix_test, vals_test, TensorShape( { i64TestSize, 1, 1 } ) );

   //auto test_address       = Placeholder( root.WithOpName("test_address"), DT_STRING );
   //auto test_address_shape = Placeholder::Shape( PartialTensorShape( { -1, 1, 1 } ) );
   //auto test_address_ix    = Placeholder( root.WithOpName("test_address_ix"), DT_INT64, test_address_shape );
   //auto test_zip           = Placeholder( root.WithOpName("test_zip"), DT_FLOAT, Placeholder::Shape( PartialTensorShape( { -1,1 } ) ) );
   //auto ref_address        = Placeholder( root.WithOpName("ref_address"), DT_STRING );
   //auto ref_address_shape  = Placeholder::Shape( PartialTensorShape( { -1, 1, 1 } ) );
   //auto ref_address_ix     = Placeholder( root.WithOpName("ref_address_ix"), DT_INT64, ref_address_shape );
   //auto ref_zip            = Placeholder( root.WithOpName("ref_zip"), DT_FLOAT, Placeholder::Shape( PartialTensorShape( { -1, kstN } ) ) );

   //std::cout << "Test Street Size = " << sRepeatEntry.size() << "\n"
   //          << "Test Size        = " << sRepeatEntry[ 0 ].size() * kstN << "\n"
   //          << "dimsize ix       = " << oSparse_Refs_Set.indices().dim_size(1) << "\n";


   // Declaration of zip distance calculation 
   //auto zip_dist = Square( root, Subtract( root, ref_zip, test_zip ) );

   // Declaration of edit distance of address
   //auto address_dist = EditDistance( root
   //                        , test_address_ix
   //                        , test_address
   //                        , {kstN,1,1}// {static_cast<int64_t>(1),3}//test_address_shape
   //                        , oSparse_Refs_Set.indices()
   //                        , oSparse_Refs_Set.values()
   //                        , {kstN,1,1} //ref_address_shape
   //                        , EditDistance::Normalize(true) );

   //auto zip_max = Gather( root, Squeeze( root, zip_dist ), ArgMax( root, zip_dist, 1 ) );
   //auto zip_min = Gather( root, Squeeze( root, zip_dist ), ArgMin( root, zip_dist, 1 ) );
   //auto zip_sim = Div( root, Subtract( root, zip_max, zip_dist ), Subtract( root, zip_max, zip_min ) );
   //auto address_sim = Subtract( root, 1, address_dist );

   //const double address_weight = 0.5;
   //const double zip_weight = 1.0 - address_weight;
   //auto weighted_sim = Add( root, MatMul( root, address_weight, address_sim, MatMul::TransposeA(true) ), MatMul( root, zip_weight, zip_sim ) );
   //auto top_match_index = ArgMax( root.WithOpName("top_match_index"), weighted_sim, 1 );



// {oSparse_Test_Set, zip_test, oSparse_Refs_Set, zip_refs}
   //ClientSession::FeedType inputs = {
   //                                     {test_address_ix, oSparse_Test_Set.indices()}
   //                                   , {test_address   , oSparse_Test_Set.values() }
   //                                   , {ref_address_ix , oSparse_Refs_Set.indices()}
   //                                   , {ref_address    , oSparse_Refs_Set.values() }
   //                                   , {test_zip       , zip_test                  }
   //                                   , {ref_zip        , zip_refs                  }
   //                                 };
   //
   //const std::vector<Output> outputs = {address_dist};

   //session.Run( inputs, outputs, &output_tensor );
 //  cout << output_tensor[ 0 ].SummarizeValue(1);


//   Tensor hypho2_ix( DT_INT64, TensorShape( { static_cast<int64_t>( 4 ), 3 } ) );
//   Tensor hypho2_vals( DT_STRING, TensorShape( {static_cast<int64_t>( 4 )} ) );
////   hypho2_vals.flat<string>().setConstant("bear");
//   makeIndex( {"bear"}, hypho2_ix   );
//   makeChar ( {"bear"}, hypho2_vals );
//
//   Tensor truth2_ix( DT_INT64, TensorShape( { static_cast<int64_t>( 5 ), 3 } ) );
//   Tensor truth2_vals( DT_STRING, TensorShape( {static_cast<int64_t>( 5 )} ) );
////   truth2_vals.flat<string>().setConstant( "beers" );
//
//   makeIndex( { "beers" }, truth2_ix );
//   makeChar ( { "beers" }, truth2_vals );
//
//      // Declaration of edit distance of address
//   auto address_dist = EditDistance( root
//                           , hypho2_ix
//                           , hypho2_vals
//                           , {3,1,1}// {static_cast<int64_t>(1),3}//test_address_shape
//                           , truth2_ix
//                           , truth2_vals
//                           , {3,1,1} //ref_address_shape
//                           , EditDistance::Normalize(false) );
//
//   //ClientSession::FeedType inputs = {
//   //                                     {test_address_ix, oSparse_Test_Set.indices()}
//   //                                   , {test_address   , oSparse_Test_Set.values() }
//   //                                   , {ref_address_ix , oSparse_Refs_Set.indices()}
//   //                                   , {ref_address    , oSparse_Refs_Set.values() }
//   //                                   , {test_zip       , zip_test                  }
//   //                                   , {ref_zip        , zip_refs                  }
//   //                                 };
//   //
//
//   auto address_sim = Subtract( root, 1, truth2_ix );
//
//   LOG( INFO ) << hypho2_ix.SummarizeValue( 200 )<< "\n";
//   LOG( INFO ) << hypho2_vals.SummarizeValue( 200 )<< "\n";
//try
//{
//   session.Run( {}, {address_sim}, &output_tensor );
//}
//catch( std::exception& roException )
//{
//
//}
//return 0;
}