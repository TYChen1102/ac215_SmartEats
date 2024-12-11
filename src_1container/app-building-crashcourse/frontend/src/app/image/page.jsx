'use client';

import { useState } from 'react';
import ImageClassification from '@/components/image/ImageClassification';
import ImageHistorySidebar from '@/components/image/ImageHistorySidebar';

export default function ImagePage() {
    const [imageHistory, setImageHistory] = useState([]);
    const [image, setImage] = useState(null);
    const [prediction, setPrediction] = useState(null);

    const addImageHistory = (image, prediction) => {
        setImageHistory([...imageHistory, { image, imageName: image.name, prediction }]);
    };

    const clearImageHistory = () => {
        setImageHistory([]);
        setImage(null);
        setPrediction(null);
    };

    return (
        <div className="flex min-h-screen">
            <ImageHistorySidebar
                imageHistory={imageHistory}
                setImage={setImage}
                setPrediction={setPrediction}
                clearImageHistory={clearImageHistory} // Pass clear function
            />

            <div className="flex-1 container mx-auto max-w-3xl pt-20 pb-12 px-4">
                <div className="mb-8">
                    <h1 className="text-3xl md:text-4xl font-bold font-montserrat" style={{ color: '#A41034' }}>
                        Image Classification
                    </h1>
                    <p className="text-gray-600 mt-2">
                        Upload an image to classify its contents using AI
                    </p>
                </div>

                <ImageClassification
                    addImageHistory={addImageHistory}
                />
            </div>
        </div>
    );
}