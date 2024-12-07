'use client'

import TodoComponent from '@/components/todo/Todo';

export default function TodoPage() {
    return (
        <div className="min-h-screen pt-20 pb-12 px-4">
            <div className="container mx-auto max-w-4xl">
                {/* Header */}
                <div className="mb-8">
                    <h1 className="text-3xl md:text-4xl font-bold text-red-600 font-montserrat" style={{ color: '#A41034' }}>
                      Health and Diet Planner
                    </h1>
                    <p className="text-gray-600 mt-2" >
                    Keep track of your health tasks and diet plans. Stay organized, exercise regularly, and maintain a balanced diet for optimal wellness!
                    </p>
                </div>

                {/* Todo Component */}
                <TodoComponent />
            </div>
        </div>
    );
}