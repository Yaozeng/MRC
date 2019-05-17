CUDA_VISIBLE_DEVICES=0

paragraph_extraction ()
{
    SOURCE_DIR=$1
    TARGET_DIR=$2
    echo "Start paragraph extraction, this may take a few hours"
    echo "Source dir: $SOURCE_DIR"
    echo "Target dir: $TARGET_DIR"
    mkdir -p $TARGET_DIR/trainset
    mkdir -p $TARGET_DIR/devset
    mkdir -p $TARGET_DIR/test1set

    echo "Processing trainset"
    cat $SOURCE_DIR/train_preprocessed/trainset/search.train.json | python paragraph_extraction.py train \
            > $TARGET_DIR/trainset/search.train.json
    cat $SOURCE_DIR/train_preprocessed/trainset/zhidao.train.json | python paragraph_extraction.py train \
            > $TARGET_DIR/trainset/zhidao.train.json

    echo "Processing devset"
    cat $SOURCE_DIR/dev_preprocessed/devset/search.dev.json | python paragraph_extraction.py dev \
            > $TARGET_DIR/devset/search.dev.json
    cat $SOURCE_DIR/dev_preprocessed/devset/zhidao.dev.json | python paragraph_extraction.py dev \
            > $TARGET_DIR/devset/zhidao.dev.json

    echo "Processing testset"
    cat $SOURCE_DIR/test1_preprocessed/test1set/search.test1.json | python paragraph_extraction.py test \
            > $TARGET_DIR/test1set/search.test1.json
    cat $SOURCE_DIR/test1_preprocessed/test1set/zhidao.test1.json | python paragraph_extraction.py test \
            > $TARGET_DIR/test1set/zhidao.test1.json
    echo "Paragraph extraction done!"
}


PROCESS_NAME="$1"
case $PROCESS_NAME in
    --para_extraction)
    # Start paragraph extraction
    if [ ! -d ./data/preprocessed ]; then
        echo "Please download the preprocessed data first (See README - Preprocess)"
        exit 1
    fi
    paragraph_extraction ./data/preprocessed ./data/extracted
    ;;
    --prepare|--train|--evaluate|--predict)
        # Start Paddle baseline
        python run.py $@
    ;;
    *)
        echo $"Usage: $0 {--para_extraction|--prepare|--train|--evaluate|--predict}"
esac
