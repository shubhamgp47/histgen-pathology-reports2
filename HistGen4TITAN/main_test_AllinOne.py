import torch
import argparse
import numpy as np
from modules.tokenizers_from14 import Tokenizer, MedicalReportTokenizer
from modules.dataloaders import TitanR2DataLoader  # Changed from R2DataLoader
from modules.metrics import compute_scores
#from modules.tester_AllinOne_sg import Tester
from modules.tester_AllinOne import Tester
from modules.loss import compute_loss
from models.histgen_model import HistGenTitanModel  # Changed from HistGenModel


def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings - MODIFIED for TITAN embeddings
    parser.add_argument('--slide_embedding_dir', type=str, required=True, 
                       help='Directory containing TITAN slide embedding files (.pt)')
    parser.add_argument('--ann_path', type=str, default='data/annotation.json', 
                       help='Path to annotation file')
    parser.add_argument('--embedding_format', type=str, default='pt', 
                       choices=['pt', 'npy', 'h5'], help='Format of embedding files')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='wsi_report', 
                       choices=['iu_xray', 'mimic_cxr', 'wsi_report'], help='Dataset to be used')
    parser.add_argument('--max_seq_length', type=int, default=100, 
                       help='Maximum sequence length of reports')
    parser.add_argument('--num_workers', type=int, default=0, 
                       help='Number of workers for dataloader')
    parser.add_argument('--batch_size', type=int, default=1, 
                       help='Number of samples per batch')
    parser.add_argument('--threshold', type=int, default=3, 
                       help='the cut off frequency for the words.')
    
    # Model settings - UPDATED for TITAN
    parser.add_argument('--model_name', type=str, default='histgen_titan', 
                       choices=['histgen_titan'], help='Model used for experiment')
    
    # TITAN-specific model settings - NEW
    parser.add_argument('--titan_embedding_dim', type=int, default=768, 
                       help='Dimension of TITAN slide embeddings')
    parser.add_argument('--projection_dim', type=int, default=512, 
                       help='Dimension of projection layer output')

    # Transformer settings
    parser.add_argument('--d_vf', type=int, default=512, 
                   help='Dimension of visual features (projection_dim for TITAN)')
    parser.add_argument('--d_model', type=int, default=512, 
                       help='Dimension of Transformer')
    parser.add_argument('--d_ff', type=int, default=512, 
                       help='Dimension of FFN')
    parser.add_argument('--num_heads', type=int, default=8, 
                       help='Number of heads in Transformer')
    parser.add_argument('--num_layers', type=int, default=3, 
                       help='Number of layers of Transformer')
    parser.add_argument('--dropout', type=float, default=0.1, 
                       help='Dropout rate of Transformer')
    parser.add_argument('--logit_layers', type=int, default=1, 
                       help='Number of logit layers')
    parser.add_argument('--bos_idx', type=int, default=0, 
                       help='Index of <bos>')
    parser.add_argument('--eos_idx', type=int, default=0, 
                       help='Index of <eos>')
    parser.add_argument('--pad_idx', type=int, default=0, 
                       help='Index of <pad>')
    parser.add_argument('--use_bn', type=int, default=0, 
                       help='Whether to use batch normalization')
    parser.add_argument('--drop_prob_lm', type=float, default=0.1, 
                       help='Dropout rate of output layer')

    # REMOVED: Original HistGen-specific parameters
    # --visual_extractor, --visual_extractor_pretrained, --d_vf
    # --topk, --cmm_size, --cmm_dim, --region_size, --prototype_num
    # --threshold

    # Sample related settings (kept)
    parser.add_argument('--sample_method', type=str, default='beam_search', 
                       help='Sample methods to generate reports')
    parser.add_argument('--beam_size', type=int, default=3, 
                       help='Beam size for beam search')
    parser.add_argument('--temperature', type=float, default=1.0, 
                       help='Temperature for sampling')
    parser.add_argument('--sample_n', type=int, default=1, 
                       help='Sample number per image')
    parser.add_argument('--group_size', type=int, default=1, 
                       help='Group size')
    parser.add_argument('--output_logsoftmax', type=int, default=1, 
                       help='Whether to output probabilities')
    parser.add_argument('--decoding_constraint', type=int, default=0, 
                       help='Whether to use decoding constraint')
    parser.add_argument('--block_trigrams', type=int, default=1, 
                       help='Whether to use block trigrams')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, 
                       help='Number of GPUs to be used')
    parser.add_argument('--epochs', type=int, default=40, 
                       help='Number of training epochs')
    parser.add_argument('--save_dir', type=str, default='results/histgen_titan', 
                       help='Path to save the models')
    parser.add_argument('--record_dir', type=str, default='records/', 
                       help='Path to save experiment results')
    parser.add_argument('--save_period', type=int, default=1, 
                       help='Saving period')
    parser.add_argument('--monitor_mode', type=str, default='max', 
                       choices=['min', 'max'], help='Whether to max or min the metric')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', 
                       help='Metric to be monitored')
    parser.add_argument('--early_stop', type=int, default=50, 
                       help='Patience for training')
    parser.add_argument('--log_period', type=int, default=1000, 
                       help='Logging interval (in batches)')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', 
                       help='Type of optimizer')
    parser.add_argument('--lr_ed', type=float, default=1e-4, 
                       help='Learning rate for model parameters')
    parser.add_argument('--weight_decay', type=float, default=5e-5, 
                       help='Weight decay')
    parser.add_argument('--amsgrad', type=bool, default=True, 
                       help='Whether to use AMSGrad')

    # REMOVED: --lr_ve

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', 
                       help='Type of learning rate scheduler')
    parser.add_argument('--step_size', type=int, default=10, 
                       help='Step size of learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.8, 
                       help='Gamma of learning rate scheduler')

    # Others
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    parser.add_argument('--resume', type=str, 
                       help='Whether to resume training from existing checkpoints')
    parser.add_argument('--load', type=str, required=True,
                       help='Path to pre-trained model for testing')

    args = parser.parse_args()
    return args


def main():
    # Parse arguments
    args = parse_agrs()

    # Fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    
    # Initialize tokenizer -
    tokenizer = MedicalReportTokenizer(args)
    
    # Create test dataloader - UPDATED to use TITAN dataloader
    test_dataloader = TitanR2DataLoader(args, tokenizer, split='test', shuffle=False)
    
    # Create model - UPDATED to use TITAN model
    model = HistGenTitanModel(args, tokenizer)

    # Get function handles of loss and metrics
    criterion = compute_loss
    metrics = compute_scores

    # Build tester and start testing - SAME as original
    tester = Tester(model, criterion, metrics, args, test_dataloader)
    tester.test()


if __name__ == '__main__':
    main()
