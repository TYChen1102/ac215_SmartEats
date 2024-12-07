'use client';

import Link from 'next/link';


export default function Home() {
    return (
        <div className="page-wrapper">
            {/* Hero Section */}
            <section className="hero-section py-16 bg-pink-100">
                <div className="text-center">
                    <h1 className="text-4xl font-bold mb-4">
                        Welcome to SmartEats!
                    </h1>
                    <h2 className="text-2xl font-semibold mb-4">
                        Your Personalized Calorie and Nutrition Advisor
                    </h2>
                    <p className="max-w-2xl mx-auto text-lg">
                        We're thrilled to have you on board our journey to healthier living and better nutrition management.
                        At SmartEats, we believe that personalized nutrition can transform the way you eat, think, and feel.
                        With our app, you're not just tracking calories and nutrition; 
                        you're unlocking a world of smarter eating choices tailored just for you.
                    </p>
                </div>
            </section>

            {/* Content Section */}
            <section className="content-section py-16 bg-gray-50">
                <div className="max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-8">
                    <Link href="/image" className="block">
                        <div className="feature-card p-6 bg-white shadow-lg rounded-lg hover:shadow-xl transition-shadow">
                            <h3 className="feature-card-title text-xl font-semibold mb-2">Meal image upload</h3>
                            <p className="feature-card-description text-gray-600">
                                Effortlessly enhancing your dining experience, food recognition technology conveniently delivers comprehensive 
                                nutritional insights with just a simple image upload.
                            </p>
                        </div>
                    </Link>

                    <Link href="/audio" className="block">
                        <div className="feature-card p-6 bg-white shadow-lg rounded-lg hover:shadow-xl transition-shadow">
                            <h3 className="feature-card-title text-xl font-semibold mb-2">AI chat box</h3>
                            <p className="feature-card-description text-gray-600">
                                Dive into your nutritional journey with our 24/7 AI Chat Box. 
                                Get instant advice and personalized meal suggestions to support your health and wellness goals.
                            </p>
                        </div>
                    </Link>
                </div>
            </section>
        </div>
    );
}