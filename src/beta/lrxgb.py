
from linearRegression import linreg_start, linreg_start_trainsize, linreg_start_3
from xgb import xgb_start, xgb_start_trainsize, xgb_start_3
from cache import save_path, save_path2, csv_path

# 1
# linreg_start()
# linreg_start(response_col='coarticle_num', save_path=save_path2)
# xgb_start()
# xgb_start('coarticle_num', save_path2)

# 2
# linreg_start_trainsize()
# linreg_start_trainsize(response_col='coarticle_num', save_path=save_path2)
# xgb_start_trainsize()
# xgb_start_trainsize(response_col='coarticle_num', save_path=save_path2)

# 3
linreg_start_3()
linreg_start_3(response_col='coarticle_num', save_path=save_path2)
xgb_start_3()
xgb_start_3(response_col='coarticle_num', save_path=save_path2)


print(csv_path,save_path,save_path2)