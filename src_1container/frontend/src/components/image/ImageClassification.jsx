'use client';

import { useRef, useState, useEffect } from 'react';
import { CloudUpload, CameraAlt } from '@mui/icons-material';
import DataService from '@/services/DataService';

export default function ImageClassification({ addImageHistory }) {
    const inputFile = useRef(null);
    const [image, setImage] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [weight, setWeight] = useState('');
    const [error, setError] = useState(null);

    useEffect(() => {
        return () => {
            if (image) {
                URL.revokeObjectURL(image);
            }
        };
    }, [image]);

    const handleImageUploadClick = () => {
        inputFile.current.click();
    };

    const handleImageChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            const objectUrl = URL.createObjectURL(file);
            setImage(objectUrl);
        }
    };

    const handleSubmit = async () => {
        if (!image || !weight) {
            setError("Please select an image and enter the weight.");
            return;
        }

        setError(null);

        try {
            setIsLoading(true);

            const formData = new FormData();
            formData.append("file", inputFile.current.files[0]);
            formData.append("weight", weight);

            const response = await DataService.ImageClassificationPredict(formData);
            setPrediction(response.data);

            // Add to history in the parent component
            addImageHistory(inputFile.current.files[0], response.data);
        } catch (error) {
            console.error('Error processing image and weight:', error);
            setError("Failed to process the image. Please try again.");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="space-y-6 p-4">
            {error && (
                <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
                    <span className="block sm:inline">{error}</span>
                </div>
            )}

            <div
                onClick={handleImageUploadClick}
                className={`
                    relative cursor-pointer
                    border-2 border-dashed border-gray-300 rounded-lg
                    bg-gray-50 hover:bg-gray-100 transition-colors
                    min-h-[400px] flex flex-col items-center justify-center
                    ${isLoading ? 'opacity-50 pointer-events-none' : ''}
                `}
            >
                <input
                    type="file"
                    accept="image/*"
                    capture="camera"
                    className="hidden"
                    ref={inputFile}
                    onChange={handleImageChange}
                />

                {image ? (
                    <div className="w-full h-full p-4">
                        <img
                            src={image}
                            alt="Preview"
                            className="w-full h-full object-contain rounded-lg"
                        />
                    </div>
                ) : (
                    <div className="text-center p-6">
                        <div className="flex flex-col items-center gap-4">
                            <div className="p-4 bg-red-100 rounded-full">
                                <CloudUpload className="text-red-500 w-8 h-8" />
                            </div>
                            <div className="space-y-2">
                                <p className="text-gray-700 font-semibold">
                                    Click to upload an image
                                </p>
                                <p className="text-sm text-gray-500">
                                    or drag and drop
                                </p>
                            </div>
                            <button className="mt-4 flex items-center gap-2 px-4 py-2 bg-purple-500 text-white rounded-full hover:bg-purple-600 transition-colors"
                                style={{ background: '#A41034' }}
                            >
                                <CameraAlt className="w-5 h-5" />
                                Take Photo
                            </button>
                        </div>
                    </div>
                )}

                {isLoading && (
                    <div className="absolute inset-0 bg-white bg-opacity-75 flex items-center justify-center">
                        <div className="animate-spin rounded-full h-12 w-12 border-4 border-purple-500 border-t-transparent"></div>
                    </div>
                )}
            </div>

            <div className="flex items-center gap-4 mt-4">
                <input
                    type="number"
                    placeholder="Enter weight"
                    value={weight}
                    onChange={(e) => setWeight(e.target.value)}
                    className="border border-gray-300 rounded p-2"
                />
                <button
                    onClick={handleSubmit}
                    disabled={!image || !weight}
                    className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
                    style={{ background: '#A41034' }}
                >
                    Submit
                </button>
            </div>

            {prediction && (
                <div className="bg-white rounded-lg shadow-md overflow-hidden p-6 mt-6">
                    <h3 className="text-lg font-bold">Prediction Results</h3>

                    <h4 className="mt-4 font-bold">Classification</h4>
                    <table className="min-w-full text-left">
                        <thead>
                            <tr className="bg-gray-50">
                                <th className="px-4 py-2">Property</th>
                                <th className="px-4 py-2">Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td className="px-4 py-2 border">Label</td>
                                <td className="px-4 py-2 border">{prediction.label}</td>
                            </tr>
                            <tr>
                                <td className="px-4 py-2 border">Probability</td>
                                <td className="px-4 py-2 border">{(prediction.probability * 100).toFixed(2)}%</td>
                            </tr>
                        </tbody>
                    </table>

                    <h4 className="mt-4 font-bold">Nutritional Information</h4>
                    <table className="min-w-full text-left">
                        <thead>
                            <tr className="bg-gray-50">
                                <th className="px-4 py-2">Nutrient</th>
                                <th className="px-4 py-2">Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            {Object.entries(prediction.nutritional_info).map(([key, value]) => (
                                <tr key={key}>
                                    <td className="px-4 py-2 border">{key}</td>
                                    <td className="px-4 py-2 border">{value}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>

                    <h4 className="mt-4 font-bold">Disease Risks</h4>
                    <table className="min-w-full text-left">
                        <thead>
                            <tr className="bg-gray-50">
                                <th className="px-4 py-2">Disease</th>
                                <th className="px-4 py-2">Risk Probability</th>
                            </tr>
                        </thead>
                        <tbody>
                            {Object.entries(prediction.disease_risks).map(([key, value]) => (
                                <tr key={key}>
                                    <td className="px-4 py-2 border">{key}</td>
                                    <td className="px-4 py-2 border">{(value * 100).toFixed(2)}%</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
}