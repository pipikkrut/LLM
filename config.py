CONFIG = {
    # модель (~98M параметров)
    "VOCAB_SIZE": 16000,  # тот же токенизатор, перетокенизировать данные не нужно
    "EMBEDDING_DIM": 768,
    "N_HEAD": 12,  # head_dim = 64
    "N_LAYERS": 12,
    "BLOCK_SIZE": 512,
    "DROPOUT": 0.0,

    # обучение
    "BATCH_SIZE": 8,  # микро-батч; если памяти хватает, попробуйте 16 (и GRAD_ACCUM 8)
    "GRAD_ACCUM": 16,  # 8 * 512 * 16 = 65 536 токенов на шаг
    "MAX_STEPS": 15000,  # ~1B токенов; пересчитайте по скорости (см. лог первых 100 шагов)
    "WARMUP_STEPS": 500,
    "LEARNING_RATE": 5e-4,
    "MIN_LR_RATIO": 0.1,
    "WEIGHT_DECAY": 0.1,

    # логи и чекпоинты
    "EVAL_INTERVAL": 500,
    "EVAL_ITERS": 50,
    "LOG_INTERVAL": 50,
    "CKPT_DIR": "checkpoints",
}
