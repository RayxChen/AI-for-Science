from model_attn_3 import FES
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


if __name__ == "__main__":
    config = {
        'batch_size': 80,
        'lr': 2e-4,
    }

    epochs = 200
    name = "pl_resume_3"

    logger = TensorBoardLogger('logs', name=f'{name}')

    checkpoint_callback = ModelCheckpoint(
        filepath='./ckpt/{epoch:03d}' + f'_{name}',
        period=10,
        monitor='valid_loss',
        save_top_k=1,
    )
    model = FES(config)

    trainer = Trainer(gpus=-1, resume_from_checkpoint='./ckpt/epoch=099_data_10K_attn_3_coord_dist_key_1.ckpt',
                      max_epochs=epochs, logger=logger, checkpoint_callback=checkpoint_callback,
                      )

    trainer.fit(model)
