from control import *
import argparse
if __name__=='__main__':
 p=argparse.ArgumentParser(description='Read-only R4 status. No model or torch import.');p.parse_args();print(json.dumps(status(),ensure_ascii=False,indent=2))
