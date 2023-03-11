track_version=scmt

# if [ ! -d data/track_feats/$track_version ]; then
#     mkdir -p data/track_feats/$track_version
# fi
# ls data/$track_version/ | grep c00..pkl | xargs -I {} ln -s "$PWD"/data/$track_version/{} data/track_feats/$track_version

# if [ ! -d data/track_results/$track_version ]; then
#     mkdir -p data/track_results/$track_version
# fi
# ln -s "$PWD"/data/$track_version/*.txt data/track_results/$track_version

# if [ ! -d data/truncation_rates/$track_version ]; then
#     mkdir -p data/truncation_rates/$track_version
# fi
# ln -s "$PWD"/data/$track_version/*_truncation.pkl data/truncation_rates/$track_version

# # preprocess the original data
python preprocess.py --src_root data/track_results/$track_version --dst_root data/preprocessed_data/$track_version --feat_root data/track_feats/$track_version --trun_root data/truncation_rates/$track_version