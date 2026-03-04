import logging
import os
from abc import abstractmethod

import cv2
import pandas as pd
import torch
import numpy as np

from modules.utils import generate_heatmap
from tqdm import tqdm
import logging

class BaseTester(object):
    def __init__(self, model, criterion, metric_ftns, args):
        self.args = args

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns

        self.epochs = self.args.epochs
        self.save_dir = self.args.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self._load_checkpoint(args.load)

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        raise NotImplementedError

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _load_checkpoint(self, load_path):
        load_path = str(load_path)
        self.logger.info("Loading checkpoint: {} ...".format(load_path))
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['state_dict'])



#OG
class Tester(BaseTester):
    def __init__(self, model, criterion, metric_ftns, args, test_dataloader):
        super(Tester, self).__init__(model, criterion, metric_ftns, args)
        self.test_dataloader = test_dataloader

    #XAI Experiment
    #def compute_permutation_importance_variable_length(self, model, dataloader, metric_fn, device):
        """
        Computes permutation feature importance for variable-length feature tensors (e.g., patch or embedding features).
        Does NOT accumulate feature tensors, processes one sample at a time to avoid OOM issues.
        Returns: DataFrame with aggregated importances per feature index.
        """
        '''model.eval()
        all_importances = []

        # If using multi-GPU, use main module
        if hasattr(model, 'module'):
            tokenizer = model.module.tokenizer
        else:
            tokenizer = model.tokenizer

        print("Running permutation feature importance analysis (variable-length) ...")
        with torch.no_grad():
            for images_id, image, reports_ids, reports_masks in tqdm(dataloader, desc="Samples"):
                # Ensure batch size 1
                if isinstance(image, torch.Tensor) and image.dim() == 4:
                    image = image[0]
                n_feats = image.shape[0]
                reports_ids = reports_ids[0] if reports_ids.dim() > 1 else reports_ids
                # Forward pass for original
                output = model(image.unsqueeze(0).to(device), mode='sample')
                report_pred = tokenizer.decode_batch(output.cpu().numpy())
                ground_truth = tokenizer.decode_batch(reports_ids[1:].unsqueeze(0).cpu().numpy())
                metric_base = metric_fn({0: [ground_truth[0]]}, {0: [report_pred[0]]})
                main_metric = list(metric_base.values())[0]

                # Compute importance per feature for this sample
                importances_this_sample = []
                for feat_idx in range(n_feats):
                    image_perturbed = image.clone()
                    temp = image_perturbed[feat_idx].clone()
                    image_perturbed[feat_idx] = 0
                    out_perturbed = model(image_perturbed.unsqueeze(0).to(device), mode='sample')
                    pred_perturbed = tokenizer.decode_batch(out_perturbed.cpu().numpy())
                    metric_pert = metric_fn({0: [ground_truth[0]]}, {0: [pred_perturbed[0]]})
                    pert_metric = list(metric_pert.values())[0]
                    importance = main_metric - pert_metric
                    importances_this_sample.append(importance)
                    # Free GPU memory
                    del out_perturbed, pred_perturbed, metric_pert, pert_metric
                    torch.cuda.empty_cache()
                all_importances.append(importances_this_sample)
                # Free GPU memory for the original output as well
                del output, report_pred, ground_truth, metric_base, main_metric
                torch.cuda.empty_cache()

        # Build DataFrame for aggregation and plotting (numeric data only kept—a small matrix)
        max_len = max([len(imps) for imps in all_importances])
        importance_matrix = np.full((len(all_importances), max_len), np.nan)
        for i, imps in enumerate(all_importances):
            importance_matrix[i, :len(imps)] = imps
        df = pd.DataFrame(importance_matrix)
        return df


    def xai(self):
        #Call XAI
        if self.args.xai_permutation_importance:
            df = self.compute_permutation_importance_variable_length(
                self.model,
                self.test_dataloader,
                self.metric_ftns,
                self.device
            )
            df.to_csv(os.path.join(self.save_dir, "permutation_importance_variable_length.csv"), index=False)'''

        
    def test(self):
        self.logger.info('Start to evaluate in the test set.')
        log = dict()
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res, test_ids = [], [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in tqdm(enumerate(self.test_dataloader)):
                images_id, images, reports_ids, reports_masks = images_id[0], images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                test_ids.append(images_id)
            
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            print(log)

            # Convert to pandas DataFrame
            test_res_df = pd.DataFrame(test_res, columns=['Generated Reports'])
            test_gts_df = pd.DataFrame(test_gts, columns=['Ground Truths'])

            # Create DataFrame for IDs
            test_ids_df = pd.DataFrame(test_ids, columns=['Case ID'])

            # Merge the DataFrames
            merged_df = pd.concat([test_ids_df, test_res_df, test_gts_df], axis=1)

            # Save the merged DataFrame to a CSV file
            merged_df.to_csv(os.path.join(self.save_dir, "gen_vs_gt.csv"), index=False)
            test_res_df.to_csv(os.path.join(self.save_dir, "res.csv"), index=False)
            test_gts_df.to_csv(os.path.join(self.save_dir, "gts.csv"), index=False)

            # Save evaluation metrics to results.csv
            metrics_df = pd.DataFrame([log])  # Wrap in list to make it a single-row DataFrame
            metrics_df.to_csv(os.path.join(self.save_dir, "results.csv"), index=False)

        return log
    

    #OG
    def plot(self):
        assert self.args.batch_size == 1 and self.args.beam_size == 3
        self.logger.info('Start to plot attention weights in the test set.')
        os.makedirs(os.path.join(self.save_dir, "attentions"), exist_ok=True)
        mean = torch.tensor((0.485, 0.456, 0.406))
        std = torch.tensor((0.229, 0.224, 0.225))
        mean = mean[:, None, None]
        std = std[:, None, None]

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images_id, images, reports_ids, reports_masks) in tqdm(enumerate(self.test_dataloader)):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output = self.model(images, mode='sample')
                image = torch.clamp((images[0].cpu() * std + mean) * 255, 0, 255).int().cpu().numpy()
                report = self.model.tokenizer.decode_batch(output.cpu().numpy())[0].split()
                attention_weights = [layer.src_attn.attn.cpu().numpy()[:, :, :-1].mean(0).mean(0) for layer in
                                     self.model.encoder_decoder.model.decoder.layers]
                for layer_idx, attns in enumerate(attention_weights):
                    assert len(attns) == len(report)
                    for word_idx, (attn, word) in enumerate(zip(attns, report)):
                        os.makedirs(os.path.join(self.save_dir, "attentions", "{:04d}".format(batch_idx),
                                                 "layer_{}".format(layer_idx)), exist_ok=True)

                        heatmap = generate_heatmap(image, attn)
                        cv2.imwrite(os.path.join(self.save_dir, "attentions", "{:04d}".format(batch_idx),
                                                 "layer_{}".format(layer_idx), "{:04d}_{}.png".format(word_idx, word)),
                        heatmap)
                        


    #Experiment
    '''def plotfix(self):
        import numpy as np
        import cv2
        
        assert self.args.batch_size == 1 and self.args.beam_size == 3
        self.logger.info('Start to plot attention weights in the test set.')
        os.makedirs(os.path.join(self.save_dir, "attentions"), exist_ok=True)
        mean = torch.tensor((0.485, 0.456, 0.406))
        std = torch.tensor((0.229, 0.224, 0.225))
        mean = mean[:, None, None]
        std = std[:, None, None]
        
        def generate_heatmap(image, weights):
            """
            Modified version of original generate_heatmap to handle non-square attention weights
            """
            image = image.transpose(1, 2, 0)
            height, width, _ = image.shape
            
            # Fix for non-square weights: pad to next larger square
            size = int(np.ceil(np.sqrt(weights.shape[0])))
            padded_len = size * size
            
            # Pad weights with zeros to make perfect square
            padded_weights = np.zeros(padded_len)
            padded_weights[:weights.shape[0]] = weights
            
            # Now reshape safely (preserving original logic)
            weights = padded_weights.reshape(size, size)
            weights = weights - np.min(weights)
            weights = weights / np.max(weights)
            weights = cv2.resize(weights, (width, height))
            weights = np.uint8(255 * weights)
            heatmap = cv2.applyColorMap(weights, cv2.COLORMAP_JET)
            result = heatmap * 0.5 + image * 0.5
            return result
        
        def process_attention(attn_tensor, report_len):
            """
            Process attention tensor to match report length
            attn_tensor shape: [batch, heads, src_len, tgt_len] = [1, 8, 182, 512]
            Returns: [report_len, src_len] - attention weights per word
            """
            # Average across heads and remove batch dimension
            attn_avg_heads = attn_tensor.mean(axis=1)[0]  # Shape: [182, 512]
            
            # Take only the first 'report_len' target positions  
            attn_trimmed = attn_avg_heads[:, :report_len]  # Shape: [182, report_len]
            
            # Transpose to get attention per word: [report_len, 182]
            attn_per_word = attn_trimmed.T
            
            return attn_per_word
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images_id, images, reports_ids, reports_masks) in tqdm(enumerate(self.test_dataloader)):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output = self.model(images, mode='sample')
                image = torch.clamp((images[0].cpu() * std + mean) * 255, 0, 255).int().cpu().numpy()
                report = self.model.tokenizer.decode_batch(output.cpu().numpy())[0].split()
                report_len = len(report)
                
                self.logger.info(f"Processing batch {batch_idx}: Report length = {report_len}")
                self.logger.info(f"Report words: {report}")
                
                # Extract attention from the correct location we discovered
                if hasattr(self.model.encoder_decoder.attn_mem, 'attn'):
                    attn_tensor = self.model.encoder_decoder.attn_mem.attn.cpu().numpy()
                    self.logger.info(f"Extracted attention tensor shape: {attn_tensor.shape}")
                    
                    # Process attention to match report length
                    attn_per_word = process_attention(attn_tensor, report_len)
                    self.logger.info(f"Processed attention shape: {attn_per_word.shape}")
                    
                    # Verify dimensions match (this should now pass)
                    assert attn_per_word.shape[0] == report_len, f"Attention length {attn_per_word.shape[0]} != report length {report_len}"
                    
                    # Generate visualizations for each word
                    for word_idx, (attn, word) in enumerate(zip(attn_per_word, report)):
                        os.makedirs(os.path.join(self.save_dir, "attentions", "{:04d}".format(batch_idx),
                                                "layer_attn_mem"), exist_ok=True)
                        
                        # Generate heatmap using the fixed function
                        heatmap = generate_heatmap(image, attn)
                        
                        # Save the attention visualization
                        filename = os.path.join(self.save_dir, "attentions", "{:04d}".format(batch_idx),
                                            "layer_attn_mem", "{:04d}_{}.png".format(word_idx, word))
                        cv2.imwrite(filename, heatmap)
                    
                    self.logger.info(f"Successfully generated {len(report)} attention visualizations for batch {batch_idx}")
                    
                else:
                    self.logger.error("No attention tensor found in encoder_decoder.attn_mem")
                    self.logger.error("Check if the model has the expected attention mechanism")
'''           
    # New Experiment 6 November
    # This generates images but the heatmaps are generated from image tensors and not real images hence gibberish     
    '''def plotfix(self):
        """Generate attention visualizations for test set reports"""
        import numpy as np
        import cv2
        
        assert self.args.batch_size == 1 and self.args.beam_size == 3
        self.logger.info('Start to plot attention weights in the test set.')
        os.makedirs(os.path.join(self.save_dir, "attentions"), exist_ok=True)
        
        # Normalization parameters for image visualization
        mean = torch.tensor((0.485, 0.456, 0.406))
        std = torch.tensor((0.229, 0.224, 0.225))
        mean = mean[:, None, None]
        std = std[:, None, None]
        
        def generate_heatmap(image, weights):
            """Generate attention heatmap overlay on image"""
            image = image.transpose(1, 2, 0)
            height, width, _ = image.shape
            
            # Handle non-square attention weights by padding to square
            size = int(np.ceil(np.sqrt(weights.shape[0])))
            padded_len = size * size
            padded_weights = np.zeros(padded_len)
            padded_weights[:weights.shape[0]] = weights
            
            # Reshape and normalize
            weights = padded_weights.reshape(size, size)
            weights = weights - np.min(weights)
            weights = weights / (np.max(weights) + 1e-8)  # Add epsilon to avoid division by zero
            weights = cv2.resize(weights, (width, height))
            weights = np.uint8(255 * weights)
            heatmap = cv2.applyColorMap(weights, cv2.COLORMAP_JET)
            result = heatmap * 0.5 + image * 0.5
            return result.astype(np.uint8)
        
        def extract_word_attentions(attn_tensor, report_len):
            """
            Extract attention weights per word from attention tensor.
            Input shape: [batch, heads, src_len, tgt_len]
            Output shape: [report_len, src_len]
            """
            # Average across heads and remove batch dimension
            attn_avg = attn_tensor.mean(axis=1)[0]  # [src_len, tgt_len]
            
            # Extract attention for each generated word (up to report_len)
            attn_per_word = attn_avg[:, :report_len].T  # [report_len, src_len]
            
            return attn_per_word
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images_id, images, reports_ids, reports_masks) in tqdm(enumerate(self.test_dataloader)):
                images = images.to(self.device)
                reports_ids = reports_ids.to(self.device)
                reports_masks = reports_masks.to(self.device)
                
                # Generate report
                output = self.model(images, mode='sample')
                
                # Prepare image for visualization
                image = torch.clamp((images[0].cpu() * std + mean) * 255, 0, 255).int().cpu().numpy()
                
                # Decode generated report
                report = self.model.tokenizer.decode_batch(output.cpu().numpy())[0].split()
                report_len = len(report)
                
                # Extract attention weights
                if hasattr(self.model.encoder_decoder, 'attn_mem') and hasattr(self.model.encoder_decoder.attn_mem, 'attn'):
                    attn_tensor = self.model.encoder_decoder.attn_mem.attn.cpu().numpy()
                    
                    # Process attention to get per-word weights
                    attn_per_word = extract_word_attentions(attn_tensor, report_len)
                    
                    # Verify dimensions match
                    if attn_per_word.shape[0] != report_len:
                        self.logger.warning(f"Batch {batch_idx}: Attention shape mismatch. Expected {report_len}, got {attn_per_word.shape[0]}")
                        continue
                    
                    # Create output directory
                    batch_dir = os.path.join(self.save_dir, "attentions", f"{batch_idx:04d}", "attention_weights")
                    os.makedirs(batch_dir, exist_ok=True)
                    
                    # Generate visualization for each word
                    for word_idx, (attn, word) in enumerate(zip(attn_per_word, report)):
                        heatmap = generate_heatmap(image, attn)
                        filename = os.path.join(batch_dir, f"{word_idx:04d}_{word}.png")
                        cv2.imwrite(filename, heatmap)
                    
                else:
                    self.logger.error(f"Batch {batch_idx}: Attention tensor not found in expected location")
                    self.logger.error("Model structure may have changed - check encoder_decoder.attn_mem.attn")
            
            self.logger.info(f'Attention visualization complete. Saved to {os.path.join(self.save_dir, "attentions")}')'''


class CompetitionTester(BaseTester):
    """
    Modified Tester class for competition submissions where ground truth is not available
    """
    def __init__(self, model, args, test_dataloader):
        # Initialize without criterion and metric_ftns since we won't calculate metrics
        self.args = args

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.epochs = self.args.epochs
        self.save_dir = self.args.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self._load_checkpoint(args.load)
        self.test_dataloader = test_dataloader

    def test(self):
        """
        Generate predictions for competition submission without ground truth
        """
        self.logger.info('Start to generate predictions for competition submission.')
        
        self.model.eval()
        with torch.no_grad():
            test_predictions = []
            test_ids = []
            
            for batch_idx, batch_data in tqdm(enumerate(self.test_dataloader)):
                # Handle different dataloader formats
                if len(batch_data) == 2:  # Only images_id and images (competition format)
                    images_id, images = batch_data
                    images_id = images_id[0] if isinstance(images_id, (list, tuple)) else images_id
                    images = images.to(self.device)
                elif len(batch_data) == 4:  # Full format with ground truth (fallback)
                    images_id, images, _, _ = batch_data
                    images_id = images_id[0] if isinstance(images_id, (list, tuple)) else images_id
                    images = images.to(self.device)
                else:
                    raise ValueError(f"Unexpected batch format with {len(batch_data)} elements")
                
                # Generate predictions
                output = self.model(images, mode='sample')
                #Debugging print statements
                '''token_ids = output.cpu().numpy()[0]  # Assuming batch size = 1
                print(f"[{batch_idx}] Token IDs: {token_ids}")

                decoded = self.model.tokenizer.decode(token_ids)
                print(f"[{batch_idx}] Decoded: {decoded}")'''
                #Debugging print statements ended on above line
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                
                test_predictions.extend(reports)
                test_ids.extend([images_id] if not isinstance(images_id, list) else images_id)
            
            # Create DataFrame for submission
            submission_df = pd.DataFrame({
                'case_id': test_ids,
                'generated_report': test_predictions
            })
            
            # Save predictions for submission
            submission_path = os.path.join(self.save_dir, "competition_predictions.csv")
            submission_df.to_csv(submission_path, index=False)
            
            # Also save just the predictions in the original format for compatibility
            predictions_df = pd.DataFrame(test_predictions, columns=['Generated Reports'])
            predictions_df.to_csv(os.path.join(self.save_dir, "predictions_only.csv"), index=False)
            
            self.logger.info(f'Generated {len(test_predictions)} predictions')
            self.logger.info(f'Predictions saved to: {submission_path}')
            
            return {'num_predictions': len(test_predictions)}

    def plot(self):
        """
        Generate attention visualizations (optional for competition)
        """
        assert self.args.batch_size == 1 and self.args.beam_size == 1
        self.logger.info('Start to plot attention weights.')
        os.makedirs(os.path.join(self.save_dir, "attentions"), exist_ok=True)
        mean = torch.tensor((0.485, 0.456, 0.406))
        std = torch.tensor((0.229, 0.224, 0.225))
        mean = mean[:, None, None]
        std = std[:, None, None]

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch_data in tqdm(enumerate(self.test_dataloader)):
                # Handle different batch formats
                if len(batch_data) == 2:
                    images_id, images = batch_data
                    images = images.to(self.device)
                elif len(batch_data) == 4:
                    images_id, images, _, _ = batch_data
                    images = images.to(self.device)
                
                output = self.model(images, mode='sample')
                image = torch.clamp((images[0].cpu() * std + mean) * 255, 0, 255).int().cpu().numpy()
                report = self.model.tokenizer.decode_batch(output.cpu().numpy())[0].split()
                attention_weights = [layer.src_attn.attn.cpu().numpy()[:, :, :-1].mean(0).mean(0) for layer in
                                     self.model.encoder_decoder.model.decoder.layers]
                for layer_idx, attns in enumerate(attention_weights):
                    assert len(attns) == len(report)
                    for word_idx, (attn, word) in enumerate(zip(attns, report)):
                        os.makedirs(os.path.join(self.save_dir, "attentions", "{:04d}".format(batch_idx),
                                                 "layer_{}".format(layer_idx)), exist_ok=True)

                        heatmap = generate_heatmap(image, attn)
                        cv2.imwrite(os.path.join(self.save_dir, "attentions", "{:04d}".format(batch_idx),
                                                 "layer_{}".format(layer_idx), "{:04d}_{}.png".format(word_idx, word)),
                                    heatmap)
