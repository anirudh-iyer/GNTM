import optuna
from main import init_config, main

def objective(trial: optuna.Trial):
    args = init_config()
    # suggest hyperparams
    args.learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-3)
    args.batch_size    = trial.suggest_categorical('batch_size', [150,200])
    args.enc_nh         = trial.suggest_categorical('enc_nh', [128,256])
    args.num_epoch     = trial.suggest_int('num_epoch', 5, 15)
    args.maskrate      = trial.suggest_uniform('maskrate', 0.3, 0.7)
    # if you want to tune seed_weight too:
    args.seed_weight   = trial.suggest_uniform('seed_weight', 0.1, 2.0)

    best_loss, _ = main(args, trial)
    return best_loss

if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)

    print("Best trial:")
    print("  Value: ", study.best_trial.value)
    print("  Params:")
    for k,v in study.best_trial.params.items():
        print(f"    {k}: {v}")

    # optionally plot and save:
    import optuna.visualization as vis
    vis.plot_optimization_history(study).write_html("opt_history.html")
    vis.plot_param_importances(study).write_html("param_importance.html")
