from pymor.basic import *


def main():
    logfile = "./mylog.log"
    # only one of the two should be necessary
    set_defaults({
        # "pymor.core.logger.default_handler.filename": logfile,
        "pymor.core.logger.getLogger.filename": logfile,
        })

    # setting the level seems to be necessary, because on init
    # the level 'INFO' is only set for pymor modules
    logger = getLogger("myapp", level="INFO", filename=logfile)
    logger.info("Hello from main.")

    dim = 51
    source = NumpyVectorSpace(dim)
    U = source.random(10)
    gram_schmidt(U, product=None, copy=False)


if __name__ == "__main__":
    main()
