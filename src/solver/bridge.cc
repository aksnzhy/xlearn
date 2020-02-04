#include <jni.h>
#include "bridge.h"
#include "src/solver/solver.h"
#include "src/base/timer.h"

void Java_com_inventale_coregistration_survey_providers_XLearnProvider_run(JNIEnv *env, jobject object, jobjectArray argsArray) {
    Timer timer;
    timer.tic();
    int argsCount = env->GetArrayLength(argsArray);
    typedef char *pchar;
    auto *argv = new pchar[argsCount];
    for (int i = 0; i < argsCount; i++) {
        auto jStringArg = (jstring) (env->GetObjectArrayElement(argsArray, i));
        const char *rawString = env->GetStringUTFChars(jStringArg, nullptr);
        argv[i] = new char[strlen(rawString) + 1];
        strcpy(argv[i], rawString);
        env->ReleaseStringUTFChars(jStringArg, rawString);
    }
    bool isTraining = strcmp(argv[0], "train") == 0;
    xLearn::Solver solver;
    if (isTraining) {
        printf("Start training\n");
        solver.SetTrain();
    } else {
        printf("Start prediction\n");
        solver.SetPredict();
    }
    solver.Initialize(argsCount, argv);
    solver.StartWork();
    solver.Clear();
    Color::print_info(StringPrintf("Total time cost: %.3f (sec)", timer.toc()), NOT_IMPORTANT_MSG);
}

JNIEXPORT jint JNICALL
Java_com_inventale_coregistration_survey_providers_XLearnProvider_getBestTask(JNIEnv *env, jobject object, jstring jmodel,
                                                                              jintArray tasks, jintArray keys,
                                                                              jintArray values, jstring joutput) {
    Timer timer;
    timer.tic();
    jint *taskArray = (env)->GetIntArrayElements(tasks, nullptr);
    jint *keysArray = (env)->GetIntArrayElements(keys, nullptr);
    jint *valuesArray = (env)->GetIntArrayElements(values, nullptr);
    jsize tasks_size = (env)->GetArrayLength(tasks);
    jsize facts_size = (env)->GetArrayLength(keys);
    printf("Input data was read\n");
    printf("Tasks amount: %i \n", tasks_size);
    printf("Facts amount: %i \n", facts_size);

    auto model = env->GetStringUTFChars(jmodel, nullptr);
    auto output = env->GetStringUTFChars(joutput, nullptr);

    // Create test dataset
    xLearn::DMatrix matrix;
    matrix.Reset();
    matrix.has_label = false;
    uint32 row_id = 0;
    const auto task_idx = 0;
    for (size_t i = 0; i < tasks_size; i++) {
        matrix.AddRow();
        auto task_value = (uint32) taskArray[i];
        matrix.AddNode(row_id, task_idx, task_value);
        for (size_t j = 0; j < facts_size; j++) {
            uint32 fact_idx = keysArray[i];
            uint32 fact_value = valuesArray[i];
            matrix.AddNode(row_id, fact_idx, fact_value);
        }
        row_id++;
    }

    printf("Prediction matrix was generated\n");
    xLearn::HyperParam param;
    param.model_file = model;
    param.output_file = output;
    param.is_train = false;
    param.from_file = false;
    param.test_dataset = &matrix;

    xLearn::Solver solver;
    solver.Initialize(param);
    solver.StartWork();
    std::vector<real_t> result = solver.GetResult();
    solver.Clear();
    Color::print_info(StringPrintf("Total predict time cost: %.3f (sec)", timer.toc()), false);

    // Selection of best task
    auto bestTaskIterator = std::max_element(std::begin(result), std::end(result));
    auto bestTaskPosition = taskArray[std::distance(std::begin(result), bestTaskIterator)];

    // Release resources
    (env)->ReleaseIntArrayElements(tasks, taskArray, 0);
    (env)->ReleaseIntArrayElements(keys, keysArray, 0);
    (env)->ReleaseIntArrayElements(values, valuesArray, 0);
    return bestTaskPosition;
}