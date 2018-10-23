jbsub -mem 8g -proj clf -queue x86_12h -cores 2+1 -out log/log.clf_only python train_clf.py --desc=clf_only --pre_load=False --preprocess=False --lm_coef=0 --n_gpu=1
jbsub -mem 8g -proj clf -queue x86_12h -cores 2+1 -out log/log.clf_LM python train_clf.py --desc=clf_LM --pre_load=False --preprocess=False --lm_coef=0.5 --n_gpu=1 
#jbsub -mem 8g -proj clf -queue x86_12h -cores 2+2 -out log/log.clf_LM_pretrain python train_clf.py --desc=clf_LM_pretrain --pre_load=True --preprocess=False --lm_coef=0.5 

