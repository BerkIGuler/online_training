python3 simpler_online_training.py  -i "127.0.0.1" \
                                    -p 65432 \
                                    --replay_sample_nb 100 \
                                    --device 0 \
                                    --epochs 5 \
                                    --threshold 10 \
                                    --batch_size 8 \
                                    --patience 2
