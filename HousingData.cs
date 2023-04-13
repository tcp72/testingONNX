//using Microsoft.ML.OnnxRuntime.Tensors;

//namespace testingONNX
//{
//    public class HousingData
//    {
//    }
//}

using Microsoft.ML.OnnxRuntime.Tensors;

public class HousingData //we need to make one of these for each
{
    public float depth { get; set; }
    public float length { get; set; }
    public float area_NE { get; set; }
    //public float area_NNW { get; set; }
    public float area_NW { get; set; }
    public float area_SE { get; set; }
    public float area_SW { get; set; }
    public float wrapping_B { get; set; }
    public float wrapping_H { get; set; }
    public float wrapping_W { get; set; }
    public float samplescollected_false { get; set; }
    public float samplescollected_true { get; set; }
    public float ageatdeath_A { get; set; }
    public float ageatdeath_C { get; set; }
    public float ageatdeath_I { get; set; }
    //public float ageatdeath_IN { get; set; }
    //public float ageatdeath_In { get; set; }
    public float ageatdeath_N { get; set; }

    //public float MedianIncome { get; set; }
    //public float MedianHouseAge { get; set; }
    //public float AverageNumberOfRooms { get; set; }
    //public float AverageNumberOfBedrooms { get; set; }
    //public float Population { get; set; }
    //public float AverageOccupancy { get; set; }
    //public float Latitude { get; set; }
    //public float Longitude { get; set; }

    public Tensor<float> AsTensor()
    {
        float[] data = new float[]
        {
            depth, length, area_NE,
            area_NW, area_SE, area_SW, wrapping_B, wrapping_H, wrapping_W,
            samplescollected_false, samplescollected_true, ageatdeath_A, 
            ageatdeath_C, ageatdeath_I, ageatdeath_N
        };
        int[] dimensions = new int[] { 1, 15 }; //***adjust this number to match the number of inputs we have (8 here)
        return new DenseTensor<float>(data, dimensions);
    }
}