@echo off
setlocal enabledelayedexpansion
echo Starting parameter sweep experiments...
echo.

REM Iterate through all parameter combinations
for %%m in (LightGCN) do (
    for %%d in (ml-1m) do (
        for %%p in (True False) do (
            for %%s in (True) do (
                REM Community suppression 0.2 with users_dec_perc_suppr: 1
                for %%u in (1) do (
                    echo Running: model=%%m dataset=%%d users_drop=%%u community=0.2 power_nodes=%%p use_suppression=%%s
                    "%~dp0/.venv/Scripts/python.exe" main.py --model_name %%m --dataset_name %%d --users_dec_perc_suppr %%u --community_suppression 0.2 --suppress_power_nodes_first %%p --use_suppression %%s

                    REM Check if the command was successful
                    if errorlevel 1 (
                        echo ERROR: Experiment failed for model=%%m dataset=%%d users_drop=%%u community=0.8 power_nodes=%%p use_suppression=%%s
                        echo.
                    ) else (
                        echo SUCCESS: Completed experiment for model=%%m dataset=%%d community=0.8 users_drop=%%u
                        echo.
                    )
                )
            )
        )
    )
)

echo All experiments completed!
pause