python3 simpler_online_training.py  -i "127.0.0.1" \
                                    -p 65432 \
                                    --replay_sample_nb 10 \
                                    --device 0 \
                                    --epochs 1000 \
                                    --threshold 100 \
                                    --batch_size 4 \
                                    --patience 2
