import os
import numpy as np
#from sklearn import datasets
import time
import datetime

import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat

from data_transformer import DataTransformer
import data_helpers
from svm import SVM


if __name__ == '__main__':
    # Data loading params
    tf.compat.v1.flags.DEFINE_string("train_data_file", './data/sample.csv', "Data source.")
    tf.compat.v1.flags.DEFINE_string("dev_data_file", './data/sample.csv', "Data source.")
    tf.compat.v1.flags.DEFINE_integer("num_class", 10, "number of class")

    # model builder
    tf.compat.v1.flags.DEFINE_integer('model_version', 2, 'version number of the model.(default: 1)')
    tf.compat.v1.flags.DEFINE_string('export_path_base', './serving/model', 'version number of the model.(default: /tmp/model)')

    # Training parameters
    tf.compat.v1.flags.DEFINE_integer("batch_size", 10, "Batch Size")
    tf.compat.v1.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs")
    tf.compat.v1.flags.DEFINE_integer("evaluate_every", 1, "Evaluate model on dev set after this many steps ")
    tf.compat.v1.flags.DEFINE_integer("checkpoint_every", 10, "Save model after this many steps ")
    tf.compat.v1.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store ")

    tf.compat.v1.flags.DEFINE_float("regulation_rate", 5e-4, "Number of checkpoints to store ")

    FLAGS = tf.compat.v1.flags.FLAGS

    export_path_base = FLAGS.export_path_base
    export_path = os.path.join(
          compat.as_bytes(export_path_base),
          compat.as_bytes(str(FLAGS.model_version)))
    assert not os.path.exists(export_path), \
        'Export directory already exists. Please specify a different export directory:{}'.format(export_path)

    data_transformer = DataTransformer(FLAGS.train_data_file)
    x_train, y_train  = data_transformer.fit_with_file(FLAGS.train_data_file, FLAGS.num_class)
    num_labels = FLAGS.num_class
    print(x_train.shape, y_train.shape)
    
    x_dev, y_dev = data_transformer.fit_with_file(FLAGS.dev_data_file, FLAGS.num_class)

    with tf.Graph().as_default():
        sess = tf.compat.v1.Session()
        with sess.as_default():
            svm = SVM(sequence_length=x_train.shape[1], 
                        num_classes=FLAGS.num_class, l2_reg_lambda=FLAGS.regulation_rate)
    
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.compat.v1.train.AdamOptimizer(0.1)
            grads_and_vars = optimizer.compute_gradients(svm.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    
            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.compat.v1.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.compat.v1.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.compat.v1.summary.merge(grad_summaries)
    
            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))
    
            # Summaries for loss and accuracy
            loss_summary = tf.compat.v1.summary.scalar("loss", svm.loss)
            acc_summary = tf.compat.v1.summary.scalar("accuracy", svm.accuracy)
    
            # Train Summaries
            train_summary_op = tf.compat.v1.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.compat.v1.summary.FileWriter(train_summary_dir, sess.graph)
    
            # Dev summaries
            dev_summary_op = tf.compat.v1.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.compat.v1.summary.FileWriter(dev_summary_dir, sess.graph)
    
            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
    
            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=FLAGS.num_checkpoints)
    
            # Initialize all variables
            sess.run(tf.compat.v1.global_variables_initializer())
    
            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  svm.input_x: x_batch,
                  svm.input_y: y_batch,
                }
    
                _, step, summaries, loss_value, accuracy_value = sess.run(
                    [train_op, global_step, train_summary_op, svm.loss, svm.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss_value, accuracy_value))
                train_summary_writer.add_summary(summaries, step)
    
            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  svm.input_x: x_batch,
                  svm.input_y: y_batch,
                }
                step, summaries, loss_value, accuracy_value = sess.run(
                    [global_step, dev_summary_op, svm.loss, svm.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss_value, accuracy_value))
                if writer:
                    writer.add_summary(summaries, step)
    
            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.compat.v1.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
    
            values, indices = tf.nn.top_k(svm.predictions, 1)

            #table = tf.contrib.lookup.index_to_string_table_from_tensor(tf.constant([str(i) for i in range(1)]))
            #table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1)
            #prediction_classes = table.lookup(tf.to_int64(indices))
            prediction_classes = tf.compat.v1.to_int64(indices)
            
            builder = saved_model_builder.SavedModelBuilder(export_path)
            
            classification_inputs = utils.build_tensor_info(svm.input_x)
            classification_outputs_classes = utils.build_tensor_info(prediction_classes)
            classification_outputs_scores = utils.build_tensor_info(svm.scores)
    
            classification_signature = signature_def_utils.build_signature_def(
            inputs={signature_constants.CLASSIFY_INPUTS: classification_inputs
                         },
            outputs={
                  signature_constants.CLASSIFY_OUTPUT_CLASSES:
                  classification_outputs_classes,
                  signature_constants.CLASSIFY_OUTPUT_SCORES:
                  classification_outputs_scores
             },
             method_name=signature_constants.CLASSIFY_METHOD_NAME)
    
            prediction_signature = signature_def_utils.build_signature_def(
                   inputs={'inputX': utils.build_tensor_info(svm.input_x)},
                   outputs={'predictClass': utils.build_tensor_info(svm.predictions), 
                            'scores':utils.build_tensor_info(svm.predictions)
                            },
            method_name=signature_constants.PREDICT_METHOD_NAME)
    
            legacy_init_op = tf.compat.v1.group(tf.compat.v1.tables_initializer(), name='legacy_init_op')
      
            #add the sigs to the servable
            builder.add_meta_graph_and_variables(
                    sess, [tag_constants.SERVING],
                    signature_def_map={
                        'textclassified':
                        prediction_signature,
                        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        classification_signature,
            }, main_op=tf.compat.v1.tables_initializer())
            #save it!
            builder.save(True)
            print("Saved model to {}\n".format(export_path))
    
