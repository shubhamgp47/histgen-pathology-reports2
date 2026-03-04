import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            #* Below is the code to generate the mask for the report
            #* such a mask is used to indicate the positions of actual tokens versus padding positions in a sequence.
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids']) 

    def __len__(self):
        return len(self.examples)
    
class PathologySingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = os.path.join(self.image_dir, image_id + '.pt')
        image = torch.load(image_path)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample

# Added my own class to bypass the existing logic to avoid ground truths
class CompetitionPathologyDataset(BaseDataset):
    """
    Alternative dataset class specifically for competition submissions
    Always returns only (image_id, image) regardless of available annotations
    """
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.split = split
        self.transform = transform
        self.ann_path = args.ann_path
        self.ann = json.loads(open(self.ann_path, 'r').read())
        # Create examples from image files
        self.examples = self._create_examples_from_images()

    '''def _create_examples_from_images(self):
        """Create examples list from image files"""
        examples = []
        # Load your annotation JSON file
        

        # Use only the 'test' split examples from JSON
        test_examples = self.ann.get('test', [])

        # Create a set of test ids for quick lookup
        test_ids = set([ex['id'] for ex in test_examples])
        if os.path.exists(self.image_dir):
            image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.pt')]
            for image_file in sorted(image_files):
                image_id = os.path.splitext(image_file)[0]
                if image_id in test_ids:
                    examples.append({'id': image_id})
        return examples'''
    
    def _create_examples_from_images(self):
    #Create examples list from test image IDs in annotation file
        examples = []

        # Get only the test entries from loaded annotation
        test_examples = self.ann.get('test', [])

        # Strip '.tiff' from ID for comparison
        test_ids = set([os.path.splitext(ex['id'])[0] for ex in test_examples])

        if os.path.exists(self.image_dir):
            image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.pt')]
            for image_file in sorted(image_files):
                image_id = os.path.splitext(image_file)[0]
                if image_id in test_ids:
                    examples.append({'id': image_id})
                else:
                    print(f"[INFO] Skipping unmatched image: {image_id}")
        return examples


    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = os.path.join(self.image_dir, image_id + '.pt')
        image = torch.load(image_path)
        return (image_id, image)

    def __len__(self):
        return len(self.examples)