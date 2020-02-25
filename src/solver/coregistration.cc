#include <jni.h>
#include "coregistration.h"
#include "src/solver/solver.h"
#include "src/base/timer.h"

const char *const RESULT_JAVA_CLASS = "com/inventale/coregistration/survey/providers/fm/PredictionResult";

void Java_com_inventale_coregistration_survey_providers_fm_XLearnProvider_run(JNIEnv *env, jobject object, jobjectArray argsArray) {
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
    Color::print_info(StringPrintf("Total time cost: %.6f (sec)", timer.toc()), NOT_IMPORTANT_MSG);
}

bool sortDescBySecond(const std::pair<int,float > &a, const std::pair<int,float> &b){
    return (a.second > b.second);
}

JNIEXPORT jobjectArray JNICALL
Java_com_inventale_coregistration_survey_providers_fm_XLearnProvider_predict(JNIEnv *env, jobject object,
                                                                             jstring jmodel,
                                                                             jintArray tasks, jintArray keys,
                                                                             jintArray values, jint jtopSize,
                                                                             jstring joutput) {
    Timer timer;
    timer.tic();
    Color::print_action("Reading input parameters ...");
    jint *taskArray = (env)->GetIntArrayElements(tasks, nullptr);
    jint *keysArray = (env)->GetIntArrayElements(keys, nullptr);
    jint *valuesArray = (env)->GetIntArrayElements(values, nullptr);
    jsize tasks_size = (env)->GetArrayLength(tasks);
    jsize facts_size = (env)->GetArrayLength(keys);
    Color::print_info(StringPrintf("Tasks amount: %i", tasks_size));
    Color::print_info(StringPrintf("Knowledge amount: %i", facts_size));

    auto model = env->GetStringUTFChars(jmodel, nullptr);
    auto output = env->GetStringUTFChars(joutput, nullptr);

    Color::print_action("Generating prediction matrix ...");
    xLearn::DMatrix matrix;
    matrix.Reset();
    matrix.has_label = false;
    uint32 row_id = 0;
    for (size_t i = 0; i < tasks_size; i++) {
        matrix.AddRow();
        auto task_idx = (uint32) taskArray[i];
        matrix.AddNode(row_id, task_idx, 1);
        for (size_t j = 0; j < facts_size; j++) {
            uint32 fact_idx = keysArray[j];
            uint32 fact_value = valuesArray[j];
            matrix.AddNode(row_id, fact_idx, fact_value);
        }
        row_id++;
    }
    Color::print_info(StringPrintf("Prediction matrix was generated"));
    xLearn::HyperParam param;
    param.model_file = model;
    param.output_file = output;
    param.is_train = false;
    param.from_file = false;
    param.test_dataset = &matrix;

    xLearn::Solver solver;
    solver.Initialize(param);
    solver.StartWork();
    std::vector<float> result = solver.GetResult();
    solver.Clear();
    Color::print_info(StringPrintf("Total predict time cost: %.6f (sec)", timer.toc()), false);

    // Selection of best tasks
    std::vector<std::pair<int, float> > taskToResult;
    for (int i = 0; i < result.size(); i++) {
        taskToResult.emplace_back(std::make_pair(taskArray[i], result[i]));
    }
    auto topSize = jtopSize > tasks_size ? tasks_size : jtopSize;
    std::partial_sort(taskToResult.begin(), taskToResult.begin() + topSize, taskToResult.end(), sortDescBySecond);

    auto cls = (env)->FindClass(RESULT_JAVA_CLASS);
    jobjectArray out = (env)->NewObjectArray(topSize, cls, nullptr);
    jmethodID constructor = (env)->GetMethodID(cls, "<init>", "(ID)V");
    if (nullptr == constructor) return nullptr;
    for (int i = 0; i < topSize; i++) {
        auto element =  taskToResult[i];
        auto newObject = (env)->NewObject(cls, constructor, element.first, element.second);
        (env)->SetObjectArrayElement(out, i, newObject);
    }
    // Release resources
    (env)->ReleaseIntArrayElements(tasks, taskArray, 0);
    (env)->ReleaseIntArrayElements(keys, keysArray, 0);
    (env)->ReleaseIntArrayElements(values, valuesArray, 0);
    return out;
}