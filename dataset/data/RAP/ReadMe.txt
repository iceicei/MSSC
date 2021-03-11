1. rap_evaluation.m is the matlab evaluation code for instance and example based metric.

2. RAP_annotation.mat is the detail annotation information about the experimental attributes in our paper.
There are totally 7 varibals in RAP_annotation, including imagesname, position, label, partion, attribute_chinese, attribute_eng, attribute_exp.
RAP_annotation.imagesname is the 41585 image names in the RAP dataset and the image could be download in another zip file in the dataset website.
RAP_annotation.position is the absolute coordinate of the person bounding box of fullbody, head-shoulder, upperbody, lowerbody in the full image.
Each of bounding box has four points, including (x,y,w,h). The coordinates are indexed from zero.
If the part is invisibile, the corresponding coordinate are set to be are zero, such as (0, 0, 0, 0).
The attributes' name in english and chinese are shown in RAP_annotation.attribute_eng and RAP_annotation.attribute_chinese.
The "hs", "ub" and "lb" in attribute_eng mean head-shoulder, upperbody and lowerbody respectly.
RAP_annotation.attribute_exp is the shorted name of top 51 attributes in attribute_eng, which is the same as our paper.
RAP_annotation.label is the annotation for the attributes in RAP_annotation.attribute_eng.
Each row in RAP_annotation.label is an example, which is corresponding with each image in RAP_annotation.imagesname.
Each col in RAP_annotation.label is an attribute, which is corresponding with each attribute in RAP_annotation.attribute_eng.
RAP_annotation.partion is the 5 random partion for training and test, which is the same as the setting in our paper.
The RAP_annotation.paper{iter}.train is the index of trainingset and RAP_annotation.partion{iter}.test is the index of testset.



% update logs
% 2016-10-8 update the gender label, from(1,2,3) to (0,1,2), while 0 is male, 1 is femal, 2 is uncertain. 
% 2016-10-13 fix the bug: all the attributes of some person are 1.
% 2017-4-28 fix some error annotation of baldhead.