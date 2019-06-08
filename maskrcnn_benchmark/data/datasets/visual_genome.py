import os
import sys
import torch
from PIL import Image
import numpy as np

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

BOX_SCALE = 1024  # Scale at which we have the boxes

class VGDataset(torch.utils.data.Dataset):

    def __init__(self, split, img_dir, roidb_file, dict_file, image_file, transforms=None,
                filter_empty_rels=True, num_im=-1, num_val_im=5000,
                filter_duplicate_rels=True, filter_non_overlap=True):
        """
        Torch dataset for VisualGenome
        Parameters:
            split: Must be train, test, or val
            img_dir: folder containing all vg images
            roidb_file:  HDF5 containing the GT boxes, classes, and relationships
            dict_file: JSON Contains mapping of classes/relationships to words
            image_file: HDF5 containing image filenames
            filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
            filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
            num_im: Number of images in the entire dataset. -1 for all images.
            num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        """
        assert split in {'train', 'val', 'test'}
        self.split = split
        self.roidb_file = roidb_file
        self.image_file = image_file
        self.filter_non_overlap = filter_non_overlap
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'
        self.transforms = transforms

        self.ind_to_classes, self.ind_to_predicates = load_info(dict_file)

        self.split_mask, self.gt_boxes, self.gt_classes, self.relationships = load_graphs(
            self.roidb_file, self.split, num_im, num_val_im=num_val_im,
            filter_empty_rels=filter_empty_rels,
            filter_non_overlap=self.filter_non_overlap and self.is_train,
        )
        self.filenames = load_image_filenames(img_dir, image_file) # length equals to split_mask
        self.filenames = [self.filenames[i] for i in np.where(self.split_mask)[0]]


    def __getitem__(self, index):
        img = Image.open(self.filenames[index]).convert("RGB")

        # important: recover original box from BOX_SCALE
        box = self.gt_boxes[index] / BOX_SCALE * max(img.size)
        target = BoxList(box, img.size, 'xywh').convert('xyxy')
        
        target.add_field("labels", self.gt_classes[index])

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        relation = self.relationships[index].copy() # (num_rel, 3)
        if self.filter_duplicate_rels:
            # Filter out dupes!
            assert self.split == 'train'
            old_size = relation.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in relation:
                all_rel_sets[(o0, o1)].append(r)
            relation = [(k[0], k[1], np.random.choice(v)) for k,v in all_rel_sets.items()]
            relation = np.array(relation)
        
        # add relation to target
        # TODO Kaihua Tang
        # check whether it is valid
        target.add_field("relation", relation)

        return img, target, index


    def __len__(self):
        return len(self.filenames)



def load_info(dict_file):
    """
    Loads the file containing the visual genome label meanings
    """
    info = json.load(open(dict_file, 'r'))
    info['label_to_idx']['__background__'] = 0
    info['predicate_to_idx']['__background__'] = 0

    class_to_ind = info['label_to_idx']
    predicate_to_ind = info['predicate_to_idx']
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
    ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])

    return ind_to_classes, ind_to_predicates


def load_image_filenames(img_dir, image_file):
    """
    Loads the image filenames from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    Parameters:
        image_file: JSON file. Elements contain the param "image_id".
        img_dir: directory where the VisualGenome images are located
    Return: 
        List of filenames corresponding to the good images
    """
    with open(image_file, 'r') as f:
        im_data = json.load(f)

    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    fns = []
    for i, img in enumerate(im_data):
        basename = '{}.jpg'.format(img['image_id'])
        if basename in corrupted_ims:
            continue

        filename = os.path.join(img_dir, basename)
        if os.path.exists(filename):
            fns.append(filename)
    assert len(fns) == 108073
    return fns


def load_graphs(roidb_file, split, num_im, num_val_im, filter_empty_rels, filter_non_overlap):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    Parameters:
        roidb_file: HDF5
        split: (train, val, or test)
        num_im: Number of images we want
        num_val_im: Number of validation images
        filter_empty_rels: (will be filtered otherwise.)
        filter_non_overlap: If training, filter images that dont overlap.
    Return: 
        image_index: numpy array corresponding to the index of images we're using
        boxes: List where each element is a [num_gt, 4] array of ground 
                    truth boxes (x1, y1, x2, y2)
        gt_classes: List where each element is a [num_gt] array of classes
        relationships: List where each element is a [num_r, 3] array of 
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    roi_h5 = h5py.File(roidb_file, 'r')
    data_split = roi_h5['split'][:]
    split_flag = 2 if split == 'test' else 0
    split_mask = data_split == split_flag

    # Filter out images without bounding boxes
    split_mask &= roi_h5['img_to_first_box'][:] >= 0
    if filter_empty_rels:
        split_mask &= roi_h5['img_to_first_rel'][:] >= 0

    image_index = np.where(split_mask)[0]
    if num_im > -1:
        image_index = image_index[:num_im]
    if num_val_im > 0:
        if split == 'val':
            image_index = image_index[:num_val_im]
        elif split == 'train':
            image_index = image_index[num_val_im:]


    split_mask = np.zeros_like(data_split).astype(bool)
    split_mask[image_index] = True

    # Get box information
    all_labels = roi_h5['labels'][:, 0]
    all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # xywh
    assert np.all(all_boxes[:, :2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box

    im_to_first_box = roi_h5['img_to_first_box'][split_mask]
    im_to_last_box = roi_h5['img_to_last_box'][split_mask]
    im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
    im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

    # load relation labels
    _relations = roi_h5['relationships'][:]
    _relation_predicates = roi_h5['predicates'][:, 0]
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

    # Get everything by image.
    boxes = []
    gt_classes = []
    relationships = []
    for i in range(len(image_index)):
        boxes_i = all_boxes[im_to_first_box[i]:im_to_last_box[i] + 1, :]
        gt_classes_i = all_labels[im_to_first_box[i]:im_to_last_box[i] + 1]

        if im_to_first_rel[i] >= 0:
            predicates = _relation_predicates[im_to_first_rel[i]:im_to_last_rel[i] + 1]
            obj_idx = _relations[im_to_first_rel[i]:im_to_last_rel[i] + 1] - im_to_first_box[i] # range is [0, num_box)
            assert np.all(obj_idx >= 0)
            assert np.all(obj_idx < boxes_i.shape[0])
            rels = np.column_stack((obj_idx, predicates)) # (num_rel, 3), representing sub, obj, and pred
        else:
            assert not filter_empty_rels
            rels = np.zeros((0, 3), dtype=np.int32)

        if filter_non_overlap:
            assert split == 'train'
            # construct BoxList object to apply boxlist_iou method
            # give a useless (height=0, width=0)
            boxes_i_obj = BoxList(boxes_i, (0, 0), 'xywh')
            inters = boxlist_iou(boxes_i_obj, boxes_i_obj)
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np.where(rel_overs > 0.0)[0]

            if inc.size > 0:
                rels = rels[inc]
            else:
                split_mask[image_index[i]] = 0
                continue

        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        relationships.append(rels)

    return split_mask, boxes, gt_classes, relationships
