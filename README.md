# Social Explorative Attention Networks (SEAN)

This repository provides a reference implementation of *sean* as described in the paper in SIGKDD 2019:<br>
> Beyond Personalization: Social Content Recommendation for Creator Equality and Consumer Satisfaction.<br>
> Wenyi Xiao, Huan Zhao, Haojie Pan, Yangqiu Song, Vincent W. Zheng, Qiang Yang.<br>
> https://arxiv.org/abs/1905.11900 <Insert paper link>

The *sean* algorithm goes beyond personalized content recommendation by considering both content creators and consumers, which motivates us to develop a highly personalized attention based model and explore higher-order social friends.

### Basic Usage

#### Dataset download
English Files down load: please put files in the dir dataset/steemit/en/

1. processed_user_activity.json

  https://drive.google.com/file/d/1QOELfVtgGgvPFGDpEI3-ZsgKlPxZX2fb/view?usp=sharing
  
2. new_article.json

  https://drive.google.com/file/d/1YQnb44R5t75D4t8XFWDArMkAllnbUm8S/view?usp=sharing
  
3. processed_user_relation.json

  https://drive.google.com/file/d/1LEteOUNDIhY8Nc9jBHbaXKB9MxxpidSh/view?usp=sharing
  
4. processed_user_index.json

  https://drive.google.com/file/d/1SsxnVsXgjlFnUJ2uVCM0NAYm89A6Akk9/view?usp=sharing
  
#### Example - End2End version in KDD 2019
To run *sean* on Steemit-En, you can use the following command:<br/>
  
  ``python steemit_preprocessing``
  
  ``python payout.py --walk-length 10 --num-walks 3 --alpha 1``

#### Example - KEYWORD version in DMKD 2020
To run *sean* with dynamic attention on Steemit-En, you can use the following command:<br/>
  ``cd sean-key``
  
  ``python data_preprocess``
  
  ``python rs_fscore.py --walk-length 10 --num-walks 3 --alpha 1 --use-sim 0``
	
	
#### Input
The supported input format is an edgelist:

	node1_id_int node2_id_int 
		
The graph is assumed to be directed and unweighted by default. 

#### Output
The probability of clicking an unseen document by the target user.

### Citing
If you find *sean* useful for your research, please consider citing the following paper:

	@inproceedings{sean-kdd2019,
	author = {Wenyi Xiao, Huan Zhao, Haojie Pan, Yangqiu Song, Vincent W. Zheng, Qiang Yang.},
	 title = {Beyond Personalization: Social Content Recommendation for Creator Equality and Consumer Satisfaction. },
	 booktitle = {Proceedings of the 25nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
	 year = {2019}
	}
	
	@article{xiao2021social,
  	title={Social explorative attention based recommendation for content distribution platforms},
  	author={Xiao, Wenyi and Zhao, Huan and Pan, Haojie and Song, Yangqiu and Zheng, Vincent W and Yang, Qiang},
  	journal={Data Mining and Knowledge Discovery},
  	volume={35},
  	number={2},
  	pages={533--567},
  	year={2021},
  	publisher={Springer}
	}


### Miscellaneous

Please send any questions you might have about the code and/or the algorithm to <wxiaoae@cse.ust.hk>.
