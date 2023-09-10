from StyleGAN import GAN
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import Viusal, load_checkpoint
from pytorch_lightning.loggers import TensorBoardLogger


if __name__ == "__main__":
    batch_size = 200
    D_lr = 5e-5
    G_lr = 2.5e-5
    epochs = 150
    name = "from_3d"

    logger = TensorBoardLogger(f'logs', name=f'{name}')

    checkpoint_callback = ModelCheckpoint(
        filepath='./ckpt/{epoch:03d}' + f'resume_{name}',
        period=5,
    )
    model = GAN(batch_size, D_lr, G_lr, state='solid', debug=False, xrd=False, d2=False)

    model = load_checkpoint(model, './ckpt/epoch=249_3d.ckpt')

    trainer = Trainer(gpus=-1,
                      max_epochs=epochs, logger=logger, checkpoint_callback=checkpoint_callback,
                      callbacks=[Viusal(f'./resume_{name}')])

    trainer.fit(model)
