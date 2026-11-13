CONFIG = {
    # модель (~20M параметров)
    "VOCAB_SIZE": 16000,
    "EMBEDDING_DIM": 384,
    "N_HEAD": 6,  # head_dim = 64
    "N_LAYERS": 8,
    "BLOCK_SIZE": 512,
    "DROPOUT": 0.0,

    # обучение
    "BATCH_SIZE": 16,  # микро-батч; если не хватает памяти T4, поставьте 8
    "GRAD_ACCUM": 8,  # 16 * 512 * 8 = 65 536 токенов на шаг
    "MAX_STEPS": 6000,  # ~400M токенов; пересчитайте по скорости (токенов/с)
    "WARMUP_STEPS": 300,
    "LEARNING_RATE": 6e-4,
    "MIN_LR_RATIO": 0.1,
    "WEIGHT_DECAY": 0.1,

    # логи и чекпоинты
    "EVAL_INTERVAL": 500,
    "EVAL_ITERS": 50,
    "LOG_INTERVAL": 50,
    "CKPT_DIR": "checkpoints",
}
