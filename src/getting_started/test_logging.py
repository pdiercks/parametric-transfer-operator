from pymor.basic import *


def main():
    logfile = "./mylog.log"
    # both have the desired effect
    set_defaults({
        # "pymor.core.logger.default_handler.filename": logfile,
        "pymor.core.logger.getLogger.filename": logfile,
        })

    logger = getLogger("pymor")
    logger.info("Hello from main.")

    dim = 51
    source = NumpyVectorSpace(dim)
    U = source.random(10)
    gram_schmidt(U, product=None, copy=False)


if __name__ == "__main__":
    main()
