@echo off
setlocal enabledelayedexpansion

:: Define hyperparameters and their possible values
set "model_name=LightGCN ItemKNN MultiVAE"
set "dataset_name=ml-20m ... ..."
set "users_top_percent=0.01 0.05"
set "users_dec_perc_drop=0.6 0.8 1"
set "community_dropout_strength=0 1"
set "do_power_nodes_from_community=True False"
set "items_top_percent=0"
set "items_dec_perc_drop=0"

:: Loop through each combination of hyperparameters
for %%m in (%model_name%) do (
    for %%d in (%dataset_name%) do (
        for %%u in (%users_top_percent%) do (
            for %%p in (%users_dec_perc_drop%) do (
                for %%c in (%community_dropout_strength%) do (
                    for %%f in (%do_power_nodes_from_community%) do (
                        for %%i in (%items_top_percent%) do (
                            for %%j in (%items_dec_perc_drop%) do (
                                echo Running main.py with model_name=%%m, dataset_name=%%d, users_top_percent=%%u, users_dec_perc_drop=%%p, community_dropout_strength=%%c, do_power_nodes_from_community=%%f, items_top_percent=%%i, items_dec_perc_drop=%%j
                                python main.py --model_name %%m --dataset_name %%d --users_top_percent %%u --users_dec_perc_drop %%p --community_dropout_strength %%c --do_power_nodes_from_community %%f --items_top_percent %%i --items_dec_perc_drop %%j
                            )
                        )
                    )
                )
            )
        )
    )
)

endlocal