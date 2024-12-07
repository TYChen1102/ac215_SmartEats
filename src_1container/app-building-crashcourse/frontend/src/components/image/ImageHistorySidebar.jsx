'use client';

export default function ImageHistorySidebar({ imageHistory, setImage, setPrediction, clearImageHistory }) {
    return (
        <div className="flex flex-col bg-white border-r border-gray-200" style={{ width: '300px', height: '180vh' }}>
            <div className="flex items-center justify-between p-4" style={{ backgroundColor: '#A41034' }}>
                <h2 className="text-white text-lg">Image History</h2>
                <button
                    onClick={clearImageHistory}
                    className="flex items-center gap-1 px-3 py-1.5 bg-red-400 hover:bg-red-600 text-white rounded-lg transition-colors"
                >
                    Clear
                </button>
            </div>

            <div className="flex-1 overflow-y-auto">
                {imageHistory.map((entry, index) => {
                    const nutritionalInfo = entry.prediction.nutritional_info;
                    const maxRisk = Object.entries(entry.prediction.disease_risks).reduce((max, current) => 
                        current[1] > max[1] ? current : max
                    , ["None", 0]);

                    return (
                        <div
                            key={index}
                            onClick={() => {
                                setImage(entry.image);
                                setPrediction(entry.prediction);
                            }}
                            className="p-4 border-b border-gray-200 cursor-pointer hover:bg-gray-50 transition-colors"
                        >
                            <div className="flex flex-col gap-1">
                                <span className="text-gray-800 text-sm truncate">{entry.imageName}</span>
                                <span className="text-gray-500 text-xs">
                                    {`Label: ${entry.prediction.label} (${(entry.prediction.probability * 100).toFixed(2)}%)`}
                                </span>
                                <span className="text-gray-500 text-xs">
                                    {`Carbohydrates: ${nutritionalInfo.Carbohydrate}`}
                                </span>
                                <span className="text-gray-500 text-xs">
                                    {`Energy: ${nutritionalInfo.Energy}`}
                                </span>
                                <span className="text-gray-500 text-xs">
                                    {`Protein: ${nutritionalInfo.Protein}`}
                                </span>
                                <span className="text-gray-500 text-xs">
                                    {`Fat: ${nutritionalInfo.Fat}`}
                                </span>
                                <span className="text-gray-500 text-xs">
                                    {`Max Risk: ${maxRisk[0]} (${(maxRisk[1] * 100).toFixed(2)}%)`}
                                </span>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}