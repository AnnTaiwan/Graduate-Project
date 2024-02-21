import pandas as pd
import numpy as np
#  direct path
BASE_PATH = 'D:/graduate_project/src/asvpoof-2019-dataset/LA/LA'

# get the LA Train information
if __name__ == "__main__":
    train_df = pd.read_csv(f'{BASE_PATH}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt',
                        sep=" ", header=None)

    train_df.columns =['speaker_id','filename','system_id','null','class_name']
    train_df.drop(columns=['null'],inplace=True)

    train_df['filepath'] = f'{BASE_PATH}/ASVspoof2019_LA_train/flac/'+train_df.filename+'.flac'
    train_df['target'] = (train_df.class_name=='spoof').astype('int32')
    # output the info
    print("Some information about LA train:")
    print('\tLen Train', len(train_df))
    neg, pos = np.bincount(train_df['target'])
    print(f'\tpositive count:{pos} negative count:{neg}')
    total = neg + pos
    print('\tExamples:\n    \tTotal: {}\n    \tPositive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))
    # get the csv file
    train_df.to_csv('train_info.csv', index=False)