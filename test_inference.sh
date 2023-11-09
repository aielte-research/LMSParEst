python inference.py -i inference_input_data/fBm_tst_10_n-1600.json -o inference_output_data/fBm_tst_100_n-1600.csv -s fOU
python inference.py -i inference_input_data/fBm_tst_10_n-1600.tsv -o inference_output_data/fBm_tst_100_n-1600.csv -m conv1D
python inference.py -i inference_input_data/fBm_tst_100_n-1600.csv -o inference_output_data/fBm_tst_100_n-1600.csv -t
python inference.py -i inference_input_data/fBm_tst_100_n-1600.csv -o inference_output_data/fBm_tst_100_n-1600.tsv -t
python inference.py -i inference_input_data/fBm_tst_100_n-1600.csv -o inference_output_data/fBm_tst_100_n-1600.json -t
python inference.py -i inference_input_data/fBm_tst_10_n-1600_no-targets.tsv -o inference_output_data/fBm_tst_no_targets.tsv