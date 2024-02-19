import numpy as np
import rosbag
import bagpy
import pandas as pd
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from IPython import embed
import glob
import time


def convert_timestamps(rostimestamp):
    acttime, msval = str(rostimestamp).split('.')
    return time.strftime('%Y-%m-%d %H:%M:%S' + '.' + str(int(msval)), time.gmtime(int(acttime)))

def check_imu(data):
    for index, row in data.iterrows():
        if '/cam1/imu' in row['Topics']:
            return True
    return False


if __name__ == '__main__':

    for episode_path in glob.glob('/lab/html/kiran/beonav/*.bag'):
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
                print("imu data available!")
                imu_msg = data.message_by_topic('/cam1/imu')
                df_imu = pd.read_csv(imu_msg)
                df_imu['Time'] = pd.to_datetime(df_imu['Time'].apply(convert_timestamps), format="%Y-%m-%d %H:%M:%S.%f")
                df_imu = df_imu[df_imu.columns.drop(list(df_imu.filter(regex='header')))]
                df_imu = df_imu.sort_values(['Time'])
                df = pd.merge_asof(df, df_imu, on='Time', direction='nearest')

            else:
                print("no imu data available")

            #df = df.reset_index()
            df.set_index('header.seq', inplace=True)
            bridge = CvBridge()
            bag = rosbag.Bag(episode_path)


            ep_path = episode_path.split('/')[-1]
            print('/lab/tmpig10c/kiran/beonav_np/' + ep_path.split('.bag')[0] + '.npy')
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
                    'observation': np.moveaxis(img, 0, 1),
                        'state': state,
                        'action': np.array([df['linear.x'][i], df['angular.z'][i]]).astype(np.float32)
                        })
            np.save('/lab/tmpig10c/kiran/beonav_np/' + ep_path.split('.bag')[0] + '.npy', episode)
