@echo off
setlocal enabledelayedexpansion
echo Starting hyperparameter sweep experiments...
echo.

REM Iterate through all parameter combinations based on community_suppression values
for %%m in (LightGCN) do (
    for %%d in (ml-1m) do (
        for %%i in (0) do (
            for %%p in (True False) do (
                for %%s in (True) do (
                    REM Community suppression 0.2 with users_dec_perc_suppr: 1
                    for %%u in (1) do (
                        echo Running: model=%%m dataset=%%d users_drop=%%u items_drop=%%i community=0.2 power_nodes=%%p use_suppression=%%s
                        "%~dp0/.venv/Scripts/python.exe" main.py --model_name %%m --dataset_name %%d --users_dec_perc_suppr %%u --items_dec_perc_suppr %%i --community_suppression 0.2 --suppress_power_nodes_first %%p --use_suppression %%s

                        REM Check if the command was successful
                        if errorlevel 1 (
                            echo ERROR: Experiment failed for model=%%m dataset=%%d users_drop=%%u items_drop=%%i community=0.2 power_nodes=%%p use_suppression=%%s
                            echo.
                        ) else (
                            echo SUCCESS: Completed experiment for model=%%m dataset=%%d community=0.2 users_drop=%%u
                            echo.
                        )
                    )
                    REM Community suppression 0.2 with users_dec_perc_suppr: 1
                    for %%u in (1) do (
                        echo Running: model=%%m dataset=%%d users_drop=%%u community=0.2 power_nodes=%%p use_suppression=%%s
                        "%~dp0/.venv/Scripts/python.exe" main.py --model_name %%m --dataset_name %%d --users_dec_perc_suppr %%u -community_suppression 0.2 --suppress_power_nodes_first %%p --use_suppression %%s

                        REM Check if the command was successful
                        if errorlevel 1 (
                            echo ERROR: Experiment failed for model=%%m dataset=%%d users_drop=%%u community=0.4 power_nodes=%%p use_suppression=%%s
                            echo.
                        ) else (
                            echo SUCCESS: Completed experiment for model=%%m dataset=%%d community=0.4 users_drop=%%u
                            echo.
                        )
                    )
                    REM Community suppression 0.5 with users_dec_perc_suppr: 0.4 0.67 1
                    for %%u in (0.4 0.67 1) do (
                        echo Running: model=%%m dataset=%%d users_drop=%%u community=0.5 power_nodes=%%p use_suppression=%%s
                        "%~dp0/.venv/Scripts/python.exe" main.py --model_name %%m --dataset_name %%d --users_dec_perc_suppr %%u --community_suppression 0.5 --suppress_power_nodes_first %%p --use_suppression %%s

                        REM Check if the command was successful
                        if errorlevel 1 (
                            echo ERROR: Experiment failed for model=%%m dataset=%%d users_drop=%%u community=0.5 power_nodes=%%p use_suppression=%%s
                            echo.
                        ) else (
                            echo SUCCESS: Completed experiment for model=%%m dataset=%%d community=0.5 users_drop=%%u
                            echo.
                        )
                    )
                )
            )
        )
    )
)

echo All hyperparameter sweep experiments completed!
echo Total experiments run: (14 parameter combinations × 3 datasets * 3 models)
pause