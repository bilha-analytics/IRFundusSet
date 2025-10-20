'''
@bg
TODO: 
- documentation 
- 
'''
import sys 
from pathlib import Path 

base_dir = str(Path(".").resolve())
if base_dir not in sys.path: 
    sys.path.append(base_dir)

from pathlib import Path
import configparser
import argparse 


from irfundusset import IRFundusSet


def get_request_argz():
    def default_out_dir():  #TODO
        return "../../output"
            
    parser = argparse.ArgumentParser(description="Generate Integrated Retinal Fundus Dataset from 10 public datasets")  
    parser.add_argument("in_cohorts_config", type=str) 
    parser.add_argument("--out_dir", "-o", type=str, default=default_out_dir() )
    parser.add_argument("--out_img_w_size", "-w", type=int, default=32) 
    parser.add_argument("--method", "-m", type=str, default='zscore' )
    parser.add_argument("--force_regenerate", "-f",action='store_true') 
    parser.add_argument("--clahe_preprocess", "-c",action='store_true') 
    
    argz = parser.parse_args() 
    return argz 
    
    

## COMMZ
def notify_status(src, status, msg, flag_error=False): 
    status = f"{'ERR' if flag_error else 'INF'} {status}"
    print(f">>> {src:10s} [{status:7s}] {msg}") 
    

if __name__ == "__main__":
    print("STARTING")
    ## i. env/context 
    argz = get_request_argz()       
    odir = f"{argz.out_dir}__{argz.method}"
    
    ## ii. make it
    hstatus, hcollection = IRFundusSet(out_dir=odir,
                          out_img_w_size=argz.out_img_w_size,
                          in_cohorts_config=argz.in_cohorts_config, 
                          force_regenerate=argz.force_regenerate,
                          clahe_b4_harmonize=argz.clahe_preprocess,
                          method=argz.method, 
                          generate_only=True)
    
    notify_status('main.generate', hstatus, f"fp={hcollection}")
    print(f"Generated HLISTING : status={hstatus}, fp={hcollection}") 
    