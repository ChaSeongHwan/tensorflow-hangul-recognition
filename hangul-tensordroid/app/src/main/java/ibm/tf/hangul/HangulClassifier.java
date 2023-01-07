package ibm.tf.hangul;

import android.content.res.AssetManager;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.TreeMap;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;


public class HangulClassifier {

    private TensorFlowInferenceInterface tfInterface;

    private String inputName;
    private String keepProbName;
    private String outputName;
    private int imageDimension;

    private List<String> labels;
    private float[] output;
    private String[] outputNames;

    /**
     * This function will create the TensorFlow inference interface and will populate all
     * the necessary data needed for inference.
     * 이 함수는 TensorFlow 인터페이스를 생성하고 인터페이스에 필요한 모든 데이터를 채운다.
     * AssetManager는 앱의 Apk 파일에 어떤 파일을 저장하고 앱에서 그 파일을 읽을 때 사용함.
     */
    public static HangulClassifier create(AssetManager assetManager,
                                          String modelPath, String labelFile, int inputDimension,
                                          String inputName, String keepProbName,
                                          String outputName) throws IOException {

        HangulClassifier classifier = new HangulClassifier();

        // These refer to the names of the nodes we care about in the model graph.
        // 모델 그래프에서 관심을 가지는 노드의 이름을 나타낸다.
        classifier.inputName = inputName;
        classifier.keepProbName = keepProbName;
        classifier.outputName = outputName;

        // Read the labels from the given label file.
        // label 파일에서 label들을 읽어옴.
        classifier.labels = readLabels(assetManager, labelFile);

        // Create the TensorFlow interface using the specified model.
        // 지정된 모델을 사용하여 TensorFlow 인터페이스 생성.
        classifier.tfInterface = new TensorFlowInferenceInterface(assetManager, modelPath);
        int numClasses = classifier.labels.size();

        // The size (in pixels) of each dimension of the image. Each dimension should be the same
        // since this is a square image.
        // 정사각형 이미지를 사용하므로 각 치수가 동일해야 한다.
        classifier.imageDimension = inputDimension;

        // This is a list of output nodes which should be filled by the inference pass.
        classifier.outputNames = new String[] { outputName };

        // This is the output node we care about.
        classifier.outputName = outputName;

        // The float buffer where the output of the softmax/output node will be stored.
        // softmax/output 노드가 저장될 버퍼.
        classifier.output = new float[numClasses];

        return classifier;
    }

    /**
     * Use the TensorFlow model and the given pixel data to produce possible classifications.
     * TensorFlow 모델과 픽셀 데이터로 최대한 분류함.
     */
    public String[] classify(final float[] pixels) {

        // Feed the image data and the keep probability (for the dropout placeholder) to the model.
        tfInterface.feed(inputName, pixels, 1, imageDimension, imageDimension, 1);
        tfInterface.feed(keepProbName, new float[] { 1 });

        // Run inference between the input and output.
        tfInterface.run(outputNames);

        // Fetch the contents of the Tensor denoted by 'outputName', copying the contents
        // into 'output'.
        // outputName으로 표시된 내용을 output에 내용을 복사함.
        tfInterface.fetch(outputName, output);

        // Map each Float to it's index. The higher values are the only ones we care about
        // and these are unlikely to have duplicates.
        // 각 float값을 인덱스에 매핑함. 더 높은 값은 유일값이며 중복되지 않을 가능성이 높다.
        TreeMap<Float,Integer> map = new TreeMap<>();
        for (int i = 0; i < output.length; i++) {
            map.put( output[i], i );
        }

        // Let's only return the top five labels in order of confidence.
        // 신뢰도 순으로 상위 5개의 label 반환.
        Arrays.sort(output);
        String[] topLabels = new String[5];
        for (int i = output.length; i > output.length-5; i--) {
            topLabels[output.length - i] = labels.get(map.get(output[i-1]));
        }
        return topLabels;
    }

    /**
     * Read in all our labels into memory so that we can map Korean characters to confidences
     * listed in the output vectors.
     * 출력 벡터에 나열된 상위 5개의 신뢰도에 한국어 문자를 매핑할수 있도록 모든 label을 메모리로 읽음.
     */
    private static List<String> readLabels(AssetManager am, String fileName) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(am.open(fileName)));

        String line;
        List<String> labels = new ArrayList<>();
        while ((line = reader.readLine()) != null) {
            labels.add(line);
        }
        reader.close();
        return labels;
    }
}
