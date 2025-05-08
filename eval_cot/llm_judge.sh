nohup python llm_judge.py --api_key AIzaSyAbogSNYhQP1HXIgXBBGIpMQvfdfOAAc1I --file_types llama32_1b > ./log/llama32_1b_llm.log 2>&1 &
nohup python llm_judge.py --api_key AIzaSyAYmHjpShRB1A0eu4ezl1fcqJtf_AUbz-k --file_types llama32_3b > ./log/llama32_3b_llm.log 2>&1 &
nohup python llm_judge.py --api_key AIzaSyBnDOLuRCyG_auCMqIYKoel3piq9b57y38 --file_types llama32_1b_loracot > ./log/llama32_1b_loracot_llm.log 2>&1 &
nohup python llm_judge.py --api_key AIzaSyDiLNbjetIFemfJCilKS9gboVZH1PSGVjU --file_types llama32_3b_loracot > ./log/llama32_3b_loracot_llm.log 2>&1 &
