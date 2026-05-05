import os
import sys
# Simple entrypoint that runs the Python replication script
sys.path.insert(0, os.path.join(os.getcwd(), 'replication_data'))
from replication_data.Popper_Replication__py import main

if __name__ == '__main__':
    main()
