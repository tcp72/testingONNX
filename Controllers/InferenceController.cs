//using Microsoft.AspNetCore.Mvc;
//using Microsoft.ML.OnnxRuntime.Tensors;
//using Microsoft.ML.OnnxRuntime;

//namespace testingONNX.Controllers
//{
//    public class OnnxController : Controller
//    {
//        public IActionResult Index()
//        {
//            return View();
//        }
//    }
//}

using System.Collections.Generic;
using System.Linq;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace testingONNX.Controllers
{
    [ApiController]
    [Route("/score")] //type in /score to make requests to the endpoint
    public class InferenceController : ControllerBase
    {
        private InferenceSession _session;

        public InferenceController(InferenceSession session)
        {
            _session = session;
        }

        [HttpPost]
        public ActionResult Score(HousingData data)
        {
            var result = _session.Run(new List<NamedOnnxValue> //this is where we actually score our model
            {
                NamedOnnxValue.CreateFromTensor("float_input", data.AsTensor())
            });
            //Tensor<string> score = result.First().AsTensor<string>(); //changed these two from float to string

            Tensor<string> score = result.First().AsTensor<string>();
            var categories = new[] { "W", "E"};
            int predictionIndex = Array.IndexOf(score.ToArray(), score.Max());

            var prediction = new Prediction { PredictedValue = categories[predictionIndex] };
            return Ok(prediction);

            //var prediction = new Prediction { PredictedValue = score }; //.First() }; //was multiplying by 1000 but removed
            //result.Dispose();
            //return Ok(prediction);
        }
    }
}