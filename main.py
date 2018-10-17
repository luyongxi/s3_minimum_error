import tensorflow as tf
import argparse
import os
import os.path as osp

def load_batch(s3_path):

    with open("filelist.txt", 'r') as f:
        fnames = f.readlines()

    filelist = [osp.join(s3_path, 'data', fn.strip()) for fn in fnames]
    paths_queue = tf.train.string_input_producer(filelist, shuffle=False)
    reader = tf.WholeFileReader()
    keys, contents = reader.read(paths_queue)
    batch = tf.reshape(tf.decode_raw(contents, tf.float32), [1])
    batch = tf.train.shuffle_batch([batch], 1, 2, 1)

    return batch

# Please put all the "data" folder to the S3_path
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Debugging options.")
    parser.add_argument('s3_path', type=str, help='path of the S3 endpoint.')
    args = parser.parse_args()

    os.environ['AWS_REGION'] = '' 
    os.environ['S3_ENDPOINT'] = 'rook-ceph-rgw-rooks3.rook' 
    os.environ['S3_USE_HTTPS'] = "0" 
    os.environ['S3_VERIFY_SSL'] = "0"
    os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

    with tf.Graph().as_default():

        batch = load_batch(args.s3_path)
        

        with tf.Session() as sess:

            for i in range(50):
                print("loading {:d}".format(i))
                print(sess.run(batch))