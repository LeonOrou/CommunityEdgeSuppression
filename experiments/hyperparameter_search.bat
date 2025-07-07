@echo off
setlocal enabledelayedexpansion
echo Starting hyperparameter search experiments...
echo.

REM ItemKNN hyperparameter search
echo Running ItemKNN hyperparameter search...
for %%t in (1 10 25 50) do (
    for %%s in (1 5 10) do (
        for %%d in (ml-100k lastfm ml-1m) do (
            echo Running: model=ItemKNN dataset=%%d topk=%%t shrink=%%s
            "%~dp0/.venv/Scripts/python.exe" main.py --model_name ItemKNN --dataset_name %%d --item_knn_topk %%t --shrink %%s

            REM Check if the command was successful
            if errorlevel 1 (
                echo ERROR: Experiment failed for model=ItemKNN dataset=%%d topk=%%t shrink=%%s
                echo.
            ) else (
                echo SUCCESS: Completed experiment for model=ItemKNN with topk=%%t shrink=%%s
                echo.
            )
        )
    )
)


REM LightGCN hyperparameter search
echo Running LightGCN hyperparameter search...
for %%e in (128) do (
    for %%l in (4) do (
        for %%d in (ml-1m) do (
            echo Running: model=LightGCN dataset=%%d embedding_dim=%%e n_layers=%%l
            "%~dp0/.venv/Scripts/python.exe" main.py --model_name LightGCN --dataset_name %%d --embedding_dim %%e --n_layers %%l

            REM Check if the command was successful
            if errorlevel 1 (
                echo ERROR: Experiment failed for model=LightGCN dataset=ml-1m embedding_dim=%%e n_layers=%%l
                echo.
            ) else (
                echo SUCCESS: Completed experiment for model=LightGCN with embedding_dim=%%e n_layers=%%l
                echo.
            )
        )
    )
)

REM MultiVAE hyperparameter search
echo Running MultiVAE hyperparameter search...
for %%h in (1000) do (
    for %%l in (200) do (
        for %%a in (0.3) do (
            for %%b in (2048) do (
                echo Running: model=MultiVAE dataset=lastfm hidden_dimension=%%h latent_dimension=%%l anneal_cap=%%a batch_size=%%b
                "%~dp0/.venv/Scripts/python.exe" main.py --model_name MultiVAE --dataset_name lastfm --hidden_dimension %%h --latent_dimension %%l --anneal_cap %%a

                REM Check if the command was successful
                if errorlevel 1 (
                    echo ERROR: Experiment failed for model=MultiVAE dataset=ml-1m hidden_dimension=%%h latent_dimension=%%l anneal_cap=%%a
                    echo.
                ) else (
                    echo SUCCESS: Completed experiment for model=MultiVAE with hidden_dimension=%%h latent_dimension=%%l anneal_cap=%%a
                    echo.
                )
            )
        )
    )
)

echo All hyperparameter search experiments completed!
pause