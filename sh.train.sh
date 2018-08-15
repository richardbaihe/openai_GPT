jbsub -require "hname=dccxc008" python train.py --ps_hosts=dccxc008.pok.ibm.com:2223 --worker_hosts=dccxc011.pok.ibm.com:2224,dccxc015.pok.ibm.com:2225 --job_name=ps --task_index=0 &&
jbsub -require "hname=dccxc011" python train.py --ps_hosts=dccxc008.pok.ibm.com:2223 --worker_hosts=dccxc011.pok.ibm.com:2224,dccxc015.pok.ibm.com:2225 --job_name=worker --task_index=0 &&
jbsub -require "hname=dccxc015" python train.py --ps_hosts=dccxc008.pok.ibm.com:2223 --worker_hosts=dccxc011.pok.ibm.com:2224,dccxc015.pok.ibm.com:2225 --job_name=worker --task_index=1 
