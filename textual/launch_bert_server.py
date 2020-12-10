from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer


args = get_args_parser().parse_args(['-model_dir', '/scratch/multi_cased_L-12_H-768_A-12/',
                                     '-port', '5555',
                                     '-port_out', '5556',
                                     #'-max_seq_len', 'NONE',
                                     '-cpu',
                                     '-num_worker', '4',
                                     '-pooling_strategy', 'REDUCE_MEAN',
                                     '-cased_tokenization'])
server = BertServer(args)
server.start()