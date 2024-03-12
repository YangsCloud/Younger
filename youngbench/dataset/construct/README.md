### Logic

1. Young_Bench_Dataset_Model (Models to be translated to ONNXs);

2. Young_Bench_Dataset_Network (Networks with is a DAG extract from an ONNX);
    * One Model CAN HAVE (Many ONNX (encoder, decoder, ...) = Many Network);
    * (One ONNX = One Network) CAN BE EXTRACTED FROM (Many Models (Same Arch));
    * `<Network, Metric>` Pair -> Find All Model ID -> Get All Metrics JSON;
    * `<Network, Metric>` Pair => 1 Net -> N Met;

3. Hugging_Face_Info:
    * One HuggingFace Model may have many README;
    * Model SHOULD Have A README;
    * Model With No README = No Metric;
    * README Corresponding To A ONNX/SubModel Is VALID;
    * README HAS Metric Is VALID;

4. RELATIONSHIP:
    * Model -- Whole ONNX/Network | Many SubONNX/SubNetwork
    * Model -- With README | Without README
    * ONNX/Network -- With README | Without README
    * README -- With Metric | Without Metric

### Process
1. Fetch All Models (HuggingFace) -- Store in Young_Bench_Dataset_Model(YBDM) -- YBDM.Maintaining = False

2. Note All Models with READMEs and Metrics -- Store in Hugging_Face_Info(HFI) -- YBDM.Maintaining = True

3. Maintain All Metrics in HFI Until Good -- Store in YBDM
    * First Round `Rules`
    * Rest Rounds `Human`

4. (YBDM.Maintaining = True & Metrics Has Value) Can Be Used/Downloaded