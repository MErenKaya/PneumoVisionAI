using Microsoft.AspNetCore.Mvc;
using System.Net.Http.Headers;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace WebApp.Controllers
{
    [Route("Diagnosis")]
    [Route("Diagnose")] 
    public class DiagnosisController : Controller
    {
        private readonly IHttpClientFactory _httpClientFactory;

        public DiagnosisController(IHttpClientFactory httpClientFactory)
        {
            _httpClientFactory = httpClientFactory;
        }

        [HttpGet]
        public IActionResult Index()
        {
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> Index(IFormFile file)
        {
            if (file == null || file.Length == 0)
            {
                ViewBag.Error = "Lütfen geçerli bir X-Ray görüntüsü seçiniz.";
                return View();
            }

            try
            {
                using var memoryStream = new MemoryStream();
                await file.CopyToAsync(memoryStream);
                var imageBytes = memoryStream.ToArray();
                string base64Image = Convert.ToBase64String(imageBytes);
                ViewBag.UploadedImage = $"data:{file.ContentType};base64,{base64Image}";

                var client = _httpClientFactory.CreateClient();

                using var content = new MultipartFormDataContent();
                using var stream = file.OpenReadStream();
                var fileContent = new StreamContent(stream);
                fileContent.Headers.ContentType = new MediaTypeHeaderValue(file.ContentType);
                content.Add(fileContent, "file", file.FileName);

                var response = await client.PostAsync("http://127.0.0.1:8000/predict", content);

                if (!response.IsSuccessStatusCode)
                {
                    ViewBag.Error = "Yapay zeka servisi yanıt vermedi.";
                    return View();
                }

                var jsonString = await response.Content.ReadAsStringAsync();
                var result = JsonSerializer.Deserialize<PredictionResultModel>(jsonString, new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                });

                return View(result);
            }
            catch (Exception ex)
            {
                ViewBag.Error = $"Servis bağlantı hatası: {ex.Message}";
                return View();
            }
        }
    }

    public class PredictionResultModel
    {
        public bool Success { get; set; }
        public string Filename { get; set; } = string.Empty;
        public string Prediction { get; set; } = string.Empty;
        public double Confidence { get; set; }
        public ProbabilitiesModel Probabilities { get; set; } = new();

        [JsonPropertyName("heatmap_base64")]
        public string HeatmapBase64 { get; set; } = string.Empty;
    }

    public class ProbabilitiesModel
    {
        [JsonPropertyName("NORMAL")]
        public double Normal { get; set; }

        [JsonPropertyName("PNEUMONIA")]
        public double Pneumonia { get; set; }
    }
}