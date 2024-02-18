from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import time
import rosbag
import tensorflow_hub as hub
import pandas as pd
import bagpy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from IPython import embed


class BeoNavDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(1280, 720, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(37,),
                            dtype=np.float32,
                            doc='Robot state, consists of [7x robot joint angles, '
                                '2x gripper position, 1x door opening angle].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(2,),
                        dtype=np.float32,
                        doc='Robot action, consists of [7x joint velocities, '
                            '2x gripper velocities, 1x terminate episode].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                        ),
                    'has_imu': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True if the episode has IMU data available'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='/lab/html/kiran/beonav/*.bag'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def convert_timestamps(rostimestamp):
            acttime, msval = str(rostimestamp).split('.')
            return time.strftime('%Y-%m-%d %H:%M:%S' + '.' + str(int(msval)), time.gmtime(int(acttime)))
        
        def check_imu(data):
            for index, row in data.iterrows():
                if '/cam1/imu' in row['Topics']:
                    return True
            return False

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            #data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case

            data = bagpy.bagreader(episode_path)
            has_imu = check_imu(data.topic_table)

            img_msg = data.message_by_topic('/cam1/color/image_raw/compressed')
            df_img = pd.read_csv(img_msg)
            df_img['Time'] = pd.to_datetime(df_img['Time'].apply(convert_timestamps), format="%Y-%m-%d %H:%M:%S.%f")

            cmd_msg = data.message_by_topic('/cmd_vel')
            df_cmd = pd.read_csv(cmd_msg)
            df_cmd['Time'] = pd.to_datetime(df_cmd['Time'].apply(convert_timestamps), format="%Y-%m-%d %H:%M:%S.%f")
            df_cmd = df_cmd[df_cmd.columns.drop(list(df_cmd.filter(regex='header')))]

            df_img = df_img.sort_values(['Time'])
            df_cmd = df_cmd.sort_values(['Time'])
            df = pd.merge_asof(df_img, df_cmd, on='Time', direction='nearest')

            if has_imu:
                print("IMU data available!")
                imu_msg = data.message_by_topic('/cam1/imu')
                df_imu = pd.read_csv(imu_msg)
                df_imu['Time'] = pd.to_datetime(df_imu['Time'].apply(convert_timestamps), format="%Y-%m-%d %H:%M:%S.%f")
                df_imu = df_imu[df_imu.columns.drop(list(df_imu.filter(regex='header')))]
                df_imu = df_imu.sort_values(['Time'])
                df = pd.merge_asof(df, df_imu, on='Time', direction='nearest')

            else:
                print("No IMU data available")
           
            #df = df.reset_index()
            df.set_index('header.seq', inplace=True)
            bridge = CvBridge()
            bag = rosbag.Bag(episode_path)




            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for _, msg, _ in bag.read_messages(topics=['/cam1/color/image_raw/compressed']):
                img = None
                #print(df['header.seq'][i], msg.header.seq)
                i = msg.header.seq
                #embed()
                assert(df['header.stamp.secs'][i] == msg.header.stamp.secs and df['header.stamp.nsecs'][i] == msg.header.stamp.nsecs and df['header.frame_id'][i] == msg.header.frame_id)
                img = bridge.compressed_imgmsg_to_cv2(msg)
                
                if has_imu:
                    list_state = []
                    for eachcol in df.columns[12:]:
                        list_state.append(df[eachcol][i])
                    state = np.array(list_state, dtype=np.float32)
                else:
                    state = np.zeros(37, dtype=np.float32)
                
                episode.append({
                    'observation': {
                        'image': np.moveaxis(img, 0, 1),
                        'state': state,
                    },
                    'action': np.array([df['linear.x'][i], df['angular.z'][i]]).astype(np.float32),
                    'discount': 1.0,
                    'reward': float(i == (df.shape[0] - 1)),
                    'is_first': i == 0,
                    'is_last': i == (df.shape[0] - 1),
                    'is_terminal': i == (df.shape[0] - 1),
                })

            #embed()
            # create output data sample
            print("finishing: ", episode_path, " has_imu: ", has_imu)
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path,
                    'has_imu': has_imu
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)
        #episode_paths = ['/lab/html/kiran/beonav/2023-08-20-01-07-49.bag', '/lab/html/kiran/beonav/2023-08-20-01-25-41.bag']
        #episode_paths = ['/lab/html/kiran/beonav/2023-03-27-11-21-26.bag', '/lab/html/kiran/beonav/2023-06-19-19-27-05.bag']
        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            try:
                yield _parse_example(sample)
            except:
                print("did not process: ", sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

